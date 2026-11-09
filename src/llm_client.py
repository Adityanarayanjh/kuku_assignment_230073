"""Provider-agnostic LLM client: JSON output, schema validation, retries, caching, rate-limit backoff.
Swap the model with LLM_MODEL / LLM_PROVIDER in .env, or add a new subclass of LLMClient."""
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import config

log = logging.getLogger("promoforge.llm")


class LLMError(RuntimeError):
    pass


class LLMClient:
    provider = "base"

    def __init__(self, model, cache_dir=None, raw_dir=None, max_retries=3, sleep=time.sleep):
        self.model, self.max_retries, self._sleep = model, max_retries, sleep
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.raw_dir = Path(raw_dir) if raw_dir else None   # unedited model replies are kept here
        for d in (self.cache_dir, self.raw_dir):
            if d:
                d.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0
        self.errors = []

    # -- to implement per provider -------------------------------------------------
    def _call(self, prompt: str, system: str = None) -> str:
        raise NotImplementedError

    # -- shared logic -----------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str):
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
            if m:
                return json.loads(m.group(1))
            raise

    def _key(self, prompt, system):
        return hashlib.sha256(f"{self.provider}|{self.model}|{system}|{prompt}".encode()).hexdigest()

    def generate_json(self, prompt: str, validator=None, system: str = None):
        key = self._key(prompt, system)
        if self.cache_dir and (self.cache_dir / f"{key}.json").exists():
            self.cache_hits += 1
            return json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        current, last_err = prompt, None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.calls += 1
                raw = self._call(current, system)
            except Exception as exc:                       # API / network / rate limit
                msg = str(exc)
                self.errors.append({"attempt": attempt, "type": "api_error", "message": msg[:300]})
                wait = 2 ** attempt * (5 if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 1)
                log.warning("API error (attempt %s): %s; waiting %ss", attempt, msg[:120], wait)
                last_err = exc
                self._sleep(wait)
                continue
            if self.raw_dir:
                (self.raw_dir / f"{key[:16]}_try{attempt}.txt").write_text(raw, encoding="utf-8")
            try:
                data = self._extract_json(raw)
                if validator:
                    data = validator(data)
                if self.cache_dir:
                    (self.cache_dir / f"{key}.json").write_text(json.dumps(data), encoding="utf-8")
                return data
            except Exception as exc:                       # malformed JSON / schema violation
                last_err = exc
                self.errors.append({"attempt": attempt, "type": "invalid_output", "message": str(exc)[:300]})
                current = (prompt + "\n\nYour previous answer was rejected: " + str(exc)[:300] +
                           "\nReturn ONLY valid JSON that follows the schema exactly.")
        raise LLMError(f"gave up after {self.max_retries} attempts: {last_err}")


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, model, api_key, **kw):
        super().__init__(model, **kw)
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        from google import genai                    # imported lazily so the core runs without it
        from google.genai import types
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def _call(self, prompt, system=None):
        cfg = self._types.GenerateContentConfig(system_instruction=system, temperature=0.2,
                                                response_mime_type="application/json")
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        return resp.text


class FakeClient(LLMClient):
    """Scripted replies for unit tests. Never used for real results."""
    provider = "fake"

    def __init__(self, replies, **kw):
        super().__init__("fake-model", **kw)
        self.replies = list(replies)

    def _call(self, prompt, system=None):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_client(raw_dir=None):
    if config.LLM_PROVIDER == "gemini":
        return GeminiClient(config.LLM_MODEL, config.LLM_API_KEY, cache_dir=config.LLM_CACHE_DIR,
                            raw_dir=raw_dir, max_retries=config.LLM_MAX_RETRIES)
    raise LLMError(f"Unknown LLM_PROVIDER {config.LLM_PROVIDER!r}. Add a subclass of LLMClient in llm_client.py")
