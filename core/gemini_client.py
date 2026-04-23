"""Minimal Gemini client for grader use.

Targets the internal proxy pattern (Bearer auth + `model` in body), configured
via GEMINI_API_URL / GEMINI_API_TOKEN in .env. Only does text-in/text-out —
multimodal lives in the vendor copies if needed.

Grader callers override the model per-request (e.g. DeepSearchQA mandates
`gemini-2.5-flash`); the env's GEMINI_MODEL is a fallback only.
"""

from __future__ import annotations

import os
import random
import time

import requests


class GeminiClient:
    def __init__(
        self,
        api_url: str | None = None,
        api_token: str | None = None,
        default_model: str | None = None,
        timeout: int = 300,
    ):
        self.api_url = api_url or os.environ.get("GEMINI_API_URL", "")
        self.api_token = api_token or os.environ.get("GEMINI_API_TOKEN", "")
        self.default_model = default_model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = timeout
        if not self.api_url or not self.api_token:
            raise RuntimeError(
                "GeminiClient needs GEMINI_API_URL + GEMINI_API_TOKEN in env/.env"
            )

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_retries: int = 5,
    ) -> str:
        # Google AI Studio: `{base}/models/<model>:generateContent?key=<api_key>`.
        # The vendor pattern used Bearer auth, but Studio rejects that; see
        # exa-DeepBench/bench/vendor_shims/gemini_client.py for the original fix.
        use_model = model or self.default_model
        url = self.api_url.rstrip("/")
        if "/models/" not in url:
            url = f"{url}/models/{use_model}:generateContent"
        if "key=" not in url:
            url = f"{url}?key={self.api_token}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                if resp.status_code >= 400:
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                text = ""
                for cand in data.get("candidates", []) or []:
                    for part in (cand.get("content") or {}).get("parts", []) or []:
                        if isinstance(part, dict) and "text" in part:
                            text += part["text"]
                return text
            except Exception as e:
                last_err = e
                if attempt == max_retries - 1:
                    break
                time.sleep(1 + 2 ** (attempt + random.random()))
        raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")
