"""The ONLY file that talks to an AI model. Swap provider/model in config.yaml.

Features: disk cache, client-side rate limit, retry with backoff, fallback models,
strict-JSON parsing with retry, token logging.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


class MissingKeyError(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg: dict, log_path: Path | None = None):
        self.cfg = cfg
        self.cache_dir = ROOT / cfg.get("cache_dir", "cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.min_gap = 60.0 / max(1, cfg.get("requests_per_minute", 8))
        self._last_call = 0.0
        self.log_path = log_path
        self.usage = {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "failures": 0}
        key_env = cfg.get("api_key_env", "GEMINI_API_KEY")
        self.api_key = os.environ.get(key_env, "").strip()
        if not self.api_key:
            raise MissingKeyError(
                f"No API key found. Create a file called .env next to app.py containing:\n{key_env}=your-key-here\n"
                "(Get a free Gemini key at https://aistudio.google.com/apikey)")
        self.provider = cfg.get("provider", "gemini")
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=cfg.get("base_url") or None)
        self.models = [cfg["model"]] + list(cfg.get("fallback_models", []))
        self.active_model = self.models[0]

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    # ---------- raw call ----------
    def _call(self, model: str, prompt: str) -> tuple[str, dict]:
        wait = self.min_gap - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()
        if self.provider == "gemini":
            from google.genai import types
            r = self.client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(temperature=self.cfg.get("temperature", 0.2),
                                                   response_mime_type="application/json"))
            u = r.usage_metadata
            usage = {"prompt_tokens": u.prompt_token_count or 0, "output_tokens": u.candidates_token_count or 0,
                     "total_tokens": u.total_token_count or 0}
            return r.text or "", usage
        r = self.client.chat.completions.create(
            model=model, temperature=self.cfg.get("temperature", 0.2),
            messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        u = r.usage
        usage = {"prompt_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens, "total_tokens": u.total_tokens}
        return r.choices[0].message.content or "", usage

    def _call_with_retry(self, prompt: str) -> tuple[str, dict, str]:
        last_err = None
        for model in self.models[self.models.index(self.active_model):]:
            for attempt in range(self.cfg.get("max_retries", 5)):
                try:
                    text, usage = self._call(model, prompt)
                    self.active_model = model
                    return text, usage, model
                except Exception as e:  # rate limits, network errors, server errors
                    last_err = e
                    msg = str(e)
                    daily = "PerDay" in msg or "per day" in msg.lower() or "limit: 0" in msg
                    if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and daily:
                        self.log(f"  quota exhausted for {model}; switching to next fallback model")
                        break
                    if "404" in msg or "NOT_FOUND" in msg:
                        self.log(f"  model {model} not available; trying fallback")
                        break
                    delay = min(60, 5 * 2 ** attempt)
                    m = re.search(r"retry in ([\d.]+)s", msg)
                    if m:
                        delay = float(m.group(1)) + 1
                    self.log(f"  API error on {model} (attempt {attempt + 1}): {msg[:120]} -> waiting {delay:.0f}s")
                    time.sleep(delay)
        self.usage["failures"] += 1
        raise RuntimeError(f"All models failed. Last error: {last_err}")

    # ---------- public: JSON call ----------
    def json_call(self, prompt: str, validate=None, tag: str = "") -> dict:
        """Return parsed JSON. `validate(obj)` returns an error string or None. Retries on bad JSON."""
        key = hashlib.sha256((self.provider + "|" + self.models[0] + "|" + prompt).encode()).hexdigest()[:24]
        cpath = self.cache_dir / f"{key}.json"
        if cpath.exists():
            self.usage["cached_calls"] += 1
            return json.loads(cpath.read_text())["parsed"]
        attempt_prompt = prompt
        for attempt in range(3):
            text, usage, model = self._call_with_retry(attempt_prompt)
            self.usage["calls"] += 1
            for k in ("prompt_tokens", "output_tokens", "total_tokens"):
                self.usage[k] += usage.get(k, 0)
            self.log(f"  LLM {tag} model={model} tokens in={usage['prompt_tokens']} out={usage['output_tokens']}")
            err = None
            try:
                obj = json.loads(_strip_fences(text))
                err = validate(obj) if validate else None
            except json.JSONDecodeError as e:
                err = f"invalid JSON: {e}"
            if not err:
                cpath.write_text(json.dumps({"model": model, "usage": usage, "raw": text, "parsed": obj}, indent=1))
                return obj
            self.log(f"  bad output ({err}); retrying")
            attempt_prompt = prompt + f"\n\nYour previous answer was rejected: {err}. Return ONLY valid JSON matching the schema."
        self.usage["failures"] += 1
        raise ValueError(f"LLM returned invalid output 3 times for {tag}")


def _strip_fences(t: str) -> str:
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(json)?", "", t).rstrip("`").strip()
    return t


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())
