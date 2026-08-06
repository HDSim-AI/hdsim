"""Model access.

One code path covers every provider that speaks the OpenAI chat API, which is most of them:
OpenAI, Ollama, vLLM, Together, OpenRouter, Groq, and university proxies. Point `base_url` at the
endpoint, set a key if it needs one, and the rest of the library does not change.

Configuration is read in this order, first match wins:

    1. arguments passed to `get_client`
    2. environment variables
    3. a `.env` file in the working directory or any parent
    4. built-in defaults

Roles let you spend money where it matters. A household member proposing a number is a cheap call;
the whole-household consensus call is worth a stronger model. Set `HDSIM_MODEL` to
run everything on one model, then override a single role if you want to.

    HDSIM_MODEL=gpt-4o-mini
    HDSIM_BASE_URL=https://api.openai.com/v1
    HDSIM_API_KEY=sk-...

    HDSIM_PERSONA_MODEL=      # persona construction
    HDSIM_MEMBER_MODEL=       # member proposals and turns
    HDSIM_MODERATOR_MODEL=    # the whole-household consensus call

Local weights are not required and not installed by default. `pip install hdsim[local]` adds a
transformers backend for offline or cluster use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROLES = ("persona", "member", "moderator")

_DEFAULTS = {
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1",
}

_ENV_LOADED = False


def load_env(start: str | Path | None = None) -> None:
    """Read a .env file from `start` or the nearest parent that has one.

    Values already present in the environment are left alone, so an exported variable always wins
    over a file. Deliberately tiny: no dependency on python-dotenv for four lines of parsing.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    here = Path(start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return


def _setting(name: str, role: str | None, override: str | None) -> str | None:
    if override:
        return override
    load_env()
    if role:
        scoped = os.environ.get(f"HDSIM_{role.upper()}_{name.upper()}")
        if scoped:
            return scoped
    return os.environ.get(f"HDSIM_{name.upper()}")


def resolve(role: str | None = None, *, model: str | None = None,
            base_url: str | None = None, api_key: str | None = None) -> dict[str, str]:
    """Work out which model and endpoint a role should use."""
    return {
        "model": _setting("model", role, model) or _DEFAULTS["model"],
        "base_url": _setting("base_url", role, base_url) or _DEFAULTS["base_url"],
        "api_key": _setting("api_key", role, api_key) or "",
    }


class Client:
    """A chat client with one method.

    Wrapping the provider SDK keeps the rest of the library free of provider details and makes the
    offline replay path a drop-in substitute.
    """

    def __init__(self, role: str | None = None, *, model: str | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 512) -> None:
        cfg = resolve(role, model=model, base_url=base_url, api_key=api_key)
        self.role = role
        self.model = cfg["model"]
        self.base_url = cfg["base_url"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = cfg["api_key"]
        self._impl = None

    def _ensure(self) -> Any:
        if self._impl is not None:
            return self._impl
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The openai package is required for live calls. Install it with "
                "`pip install hdsim` or run the offline demo with `hdsim demo`."
            ) from exc
        if not self._api_key:
            # Ollama and most local servers ignore the key but the SDK requires a value.
            self._api_key = "not-needed"
        self._impl = OpenAI(api_key=self._api_key, base_url=self.base_url)
        return self._impl

    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None,
             max_tokens: int | None = None, seed: int | None = None) -> str:
        """Send messages, return the assistant text."""
        client = self._ensure()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        response = client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()

    def __repr__(self) -> str:
        return f"Client(role={self.role!r}, model={self.model!r}, base_url={self.base_url!r})"


def get_client(role: str | None = None, **kwargs: Any) -> Client:
    """Return a client for a role, honouring role specific overrides."""
    return Client(role, **kwargs)


def check() -> dict[str, dict[str, str]]:
    """Report what each role would use. Keys are redacted.

    Useful as a first command when something is misconfigured:

        python -c "from hdsim.core.backends import check; print(check())"
    """
    out = {}
    for role in ROLES:
        cfg = resolve(role)
        out[role] = {
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "api_key": "set" if cfg["api_key"] else "not set",
        }
    return out
