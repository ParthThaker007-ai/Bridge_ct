"""HTTP client with retries, throttling, on-disk caching and a retrieval manifest.

Why caching: public APIs change, rate-limit and occasionally fail. Caching every raw response (with its
retrieval timestamp) makes the whole pipeline reproducible and lets you re-run normalisation without
hitting the network again.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..utils import ensure_dir, get_logger, sha256_text

log = get_logger()


class CachedClient:
    """GET/POST client; responses are cached under ``cache_dir`` keyed by a hash of the request.

    Parameters
    ----------
    cache_dir : directory for cached responses.
    min_interval : minimum seconds between *network* calls (cache hits are not throttled).
    timeout : per-request timeout in seconds.
    retries : urllib3 retry count for 429/5xx; ``Retry-After`` is respected.
    user_agent : identify yourself politely to public APIs.
    """

    def __init__(self, cache_dir: str | Path, min_interval: float = 0.0, timeout: float = 60.0,
                 retries: int = 5, user_agent: str = "bridge-ct/1.0 (research use)") -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_call = 0.0
        self.manifest: List[Dict[str, Any]] = []
        self.session = requests.Session()
        retry = Retry(total=retries, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=frozenset(["GET", "POST"]), respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"User-Agent": user_agent})

    # ------------------------------------------------------------------ internals
    def _key(self, method: str, url: str, params: Optional[dict], data: Optional[dict]) -> str:
        blob = json.dumps({"m": method, "u": url, "p": params, "d": data}, sort_keys=True, default=str)
        return sha256_text(blob)

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _request(self, method: str, url: str, params: Optional[dict], data: Optional[dict],
                 secret_params: Optional[dict], kind: str) -> Any:
        """Return the cached payload or fetch it. ``secret_params`` (API keys) are sent but never cached."""
        key = self._key(method, url, params, data)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
            return entry["payload"]
        self._throttle()
        send_params = {**(params or {}), **(secret_params or {})} or None
        resp = self.session.request(method, url, params=send_params, data=data, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json() if kind == "json" else resp.text
        retrieved_at = datetime.now(timezone.utc).isoformat()
        entry = {"retrieved_at": retrieved_at, "method": method, "url": url, "params": params,
                 "data_keys": sorted(data) if data else None, "payload": payload}
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entry, fh)
        tmp.replace(path)  # atomic: a crash never leaves a half-written cache file
        self.manifest.append({"url": url, "params": params, "retrieved_at": retrieved_at,
                              "cache_file": path.name, "sha256": sha256_text(json.dumps(payload, default=str))})
        return payload

    # ------------------------------------------------------------------ public
    def get_json(self, url: str, params: Optional[dict] = None, secret_params: Optional[dict] = None) -> Any:
        return self._request("GET", url, params, None, secret_params, "json")

    def post_text(self, url: str, data: dict, secret_params: Optional[dict] = None) -> str:
        return self._request("POST", url, None, data, secret_params, "text")
