"""EDGAR client unit tests — network fully mocked with respx."""
import httpx
import pytest
import respx

from citesight.config.settings import Settings
from citesight.ingest.edgar import SUBMISSIONS_URL, TICKERS_URL, EdgarClient, Filing


@pytest.fixture
def settings(tmp_path):
    return Settings(
        sec_user_agent="CiteSight tests (test@example.com)",
        data_dir=tmp_path / "data",
    )


TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

SUBMISSIONS_PAYLOAD = {
    "filings": {
        "recent": {
            "form": ["10-Q", "8-K", "10-K", "10-K"],
            "accessionNumber": [
                "0000320193-25-000001",
                "0000320193-25-000002",
                "0000320193-24-000123",
                "0000320193-23-000106",
            ],
            "filingDate": ["2025-05-01", "2025-03-01", "2024-11-01", "2023-11-03"],
            "primaryDocument": ["q.htm", "k8.htm", "aapl-10k-2024.htm", "aapl-10k-2023.htm"],
        }
    }
}


@respx.mock
def test_cik_lookup_and_filing_filter(settings):
    respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_PAYLOAD))
    respx.get(SUBMISSIONS_URL.format(cik="0000320193")).mock(
        return_value=httpx.Response(200, json=SUBMISSIONS_PAYLOAD)
    )
    client = EdgarClient(settings)
    filings = client.list_filings("aapl", forms=["10-K"], limit=1)
    assert len(filings) == 1
    f = filings[0]
    assert f.form == "10-K"
    assert f.cik == "0000320193"
    assert f.filing_date == "2024-11-01"
    assert f.primary_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-10k-2024.htm"
    )
    assert f.doc_id == "AAPL_10-K_0000320193-24-000123"


@respx.mock
def test_download_primary_is_cached(settings, tmp_path):
    respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_PAYLOAD))
    filing = Filing(
        ticker="AAPL",
        cik="0000320193",
        accession="0000320193-24-000123",
        form="10-K",
        filing_date="2024-11-01",
        primary_document="aapl-10k-2024.htm",
    )
    route = respx.get(filing.primary_url).mock(
        return_value=httpx.Response(200, content=b"<html>filing body</html>")
    )
    client = EdgarClient(settings)
    dest1 = client.download_primary(filing, tmp_path / "raw")
    dest2 = client.download_primary(filing, tmp_path / "raw")
    assert dest1 == dest2
    assert dest1.read_bytes() == b"<html>filing body</html>"
    assert route.call_count == 1  # second call served from disk


def test_missing_user_agent_raises(tmp_path):
    settings = Settings(sec_user_agent=None, data_dir=tmp_path, _env_file=None)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        EdgarClient(settings)
