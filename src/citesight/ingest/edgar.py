"""SEC EDGAR client: submissions API + primary-document download.

EDGAR etiquette: descriptive User-Agent with contact email (required),
<=10 requests/second (we default to 8), exponential backoff on 403/429/5xx.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from citesight.config.settings import Settings

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"


class Filing(BaseModel):
    ticker: str
    cik: str  # zero-padded 10 digits
    accession: str  # with dashes, e.g. 0000320193-24-000123
    form: str
    filing_date: str
    primary_document: str

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}_{self.form.replace('/', '-')}_{self.accession}"

    @property
    def primary_url(self) -> str:
        return ARCHIVES_URL.format(
            cik_int=int(self.cik),
            accession_nodash=self.accession.replace("-", ""),
            doc=self.primary_document,
        )


class RateLimiter:
    def __init__(self, max_per_sec: float) -> None:
        self.max_per_sec = max_per_sec
        self._times: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._times and now - self._times[0] > 1.0:
            self._times.popleft()
        if len(self._times) >= self.max_per_sec:
            sleep_for = 1.0 - (now - self._times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._times.append(time.monotonic())


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (403, 429) or exc.response.status_code >= 500
    return isinstance(exc, (httpx.TransportError,))


class EdgarClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.sec_user_agent:
            raise RuntimeError(
                "SEC_USER_AGENT is not set. EDGAR requires a descriptive "
                "User-Agent with a contact email (see .env.example)."
            )
        self.settings = settings
        self.client = client or httpx.Client(
            headers={
                "User-Agent": settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=30.0,
            follow_redirects=True,
        )
        self.limiter = RateLimiter(settings.edgar_max_requests_per_sec)

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        self.limiter.wait()
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp

    def cik_for_ticker(self, ticker: str) -> str:
        data = self._get(TICKERS_URL).json()
        ticker = ticker.upper()
        for entry in data.values():
            if entry["ticker"].upper() == ticker:
                return f"{int(entry['cik_str']):010d}"
        raise ValueError(f"ticker not found on EDGAR: {ticker}")

    def list_filings(
        self, ticker: str, forms: list[str], limit: int
    ) -> list[Filing]:
        """Latest `limit` filings of the given form types (most recent first)."""
        cik = self.cik_for_ticker(ticker)
        data = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        recent = data["filings"]["recent"]
        wanted = {f.upper() for f in forms}
        filings: list[Filing] = []
        for i in range(len(recent["form"])):
            if recent["form"][i].upper() not in wanted:
                continue
            filings.append(
                Filing(
                    ticker=ticker.upper(),
                    cik=cik,
                    accession=recent["accessionNumber"][i],
                    form=recent["form"][i],
                    filing_date=recent["filingDate"][i],
                    primary_document=recent["primaryDocument"][i],
                )
            )
            if len(filings) >= limit:
                break
        logger.info(
            "found %d filings for %s (forms=%s)", len(filings), ticker, sorted(wanted)
        )
        return filings

    def download_primary(self, filing: Filing, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filing.primary_document).suffix or ".htm"
        dest = dest_dir / f"{filing.doc_id}{suffix}"
        if dest.exists() and dest.stat().st_size > 0:
            logger.info("already downloaded: %s", dest.name)
            return dest
        resp = self._get(filing.primary_url)
        dest.write_bytes(resp.content)
        logger.info("downloaded %s (%.1f KB)", dest.name, len(resp.content) / 1024)
        return dest

    def close(self) -> None:
        self.client.close()
