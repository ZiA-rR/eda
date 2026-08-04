"""
llm.py
------
One function to call any of the three model providers, so the rest of the
pipeline does not care which one it is talking to.

The plan calls for scoring every candidate with three DIFFERENT models and
using their disagreement to find the cases a human needs to look at. That
only works if the three are genuinely different families, since models from
the same family share biases and would agree with each other for the wrong
reasons.

Set whichever keys you have:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GOOGLE_API_KEY

Install only what you need:
    pip install openai anthropic google-generativeai
"""

VERSION = "2026-08-03-e"

import os
import time
from typing import Optional

# Model names per provider. Update these as versions change.
# Which model names are on the free tier changes over time. If a model
# returns "limit: 0" it has no free allocation, and list_gemini_models()
# will show what the key can actually reach.
MODELS = {
    "gpt":    {"provider": "openai",    "name": "gpt-4o"},
    "claude": {"provider": "anthropic", "name": "claude-sonnet-4-5"},
    "gemini": {"provider": "google",
               "name": os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")},
}


def set_model(key: str, name: str) -> None:
    """Point a model key at a different underlying model name."""
    MODELS[key]["name"] = name
    _clients.pop(MODELS[key]["provider"], None)
    print(f"{key} -> {name}")


def list_gemini_models() -> list:
    """
    What models can this key actually use, and which support generation.
    Run this when you hit a quota error, to find one that works.
    """
    names = []
    try:
        from google import genai as g
        client = g.Client(api_key=os.environ["GOOGLE_API_KEY"])
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(m.name.replace("models/", ""))
    except ImportError:
        import google.generativeai as g
        g.configure(api_key=os.environ["GOOGLE_API_KEY"])
        for m in g.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                names.append(m.name.replace("models/", ""))

    for n in names:
        print(" ", n)
    return names

# The three used for scoring. Deliberately one from each family.
SCORING_MODELS = ["gpt", "claude", "gemini"]

_clients = {}

# Free tiers cap requests per minute as well as per day. Spacing calls out
# avoids burning retries on 429s that a short wait would have avoided.
MIN_INTERVAL = float(os.environ.get("LLM_MIN_INTERVAL", "4.0"))
_last_call = 0.0


def _rate_wait():
    global _last_call
    import time as _t
    elapsed = _t.time() - _last_call
    if elapsed < MIN_INTERVAL:
        _t.sleep(MIN_INTERVAL - elapsed)
    _last_call = _t.time()


def _get_client(provider: str):
    if provider in _clients:
        return _clients[provider]

    if provider == "openai":
        from openai import OpenAI
        _clients[provider] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "anthropic":
        import anthropic
        _clients[provider] = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "google":
        # google.generativeai is deprecated. Prefer google.genai, fall back
        # to the old package if only that is installed.
        try:
            from google import genai as new_genai
            _clients[provider] = ("new", new_genai.Client(
                api_key=os.environ["GOOGLE_API_KEY"]))
        except ImportError:
            import google.generativeai as old_genai
            old_genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            _clients[provider] = ("old", old_genai)
    else:
        raise ValueError(f"unknown provider {provider}")

    return _clients[provider]


def call_model(prompt: str,
               model: str = "claude",
               max_tokens: int = 1000,
               temperature: float = 0.0,
               retries: int = 3,
               backoff: float = 2.0) -> str:
    """
    Send a prompt, get text back.

    temperature is 0 by default. For scoring we want the same answer if we
    ask twice, otherwise the variance between models gets confused with the
    variance within a single model.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model '{model}', options: {list(MODELS)}")

    spec = MODELS[model]
    provider, name = spec["provider"], spec["name"]

    last_err = None
    for attempt in range(retries):
        try:
            _rate_wait()
            client = _get_client(provider)

            if provider == "openai":
                resp = client.chat.completions.create(
                    model=name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content

            if provider == "anthropic":
                resp = client.messages.create(
                    model=name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text

            if provider == "google":
                kind, gclient = client
                if kind == "new":
                    cfg = {"max_output_tokens": max_tokens,
                           "temperature": temperature}
                    # Gemini 2.5 and later spend output tokens on internal
                    # reasoning before answering. For structured extraction
                    # and scoring we want the answer, not the reasoning, and
                    # leaving thinking on means the budget gets eaten and
                    # .text comes back empty.
                    if any(v in name for v in ("2.5", "3.", "-latest")):
                        cfg["thinking_config"] = {"thinking_budget": 0}

                    resp = gclient.models.generate_content(
                        model=name, contents=prompt, config=cfg)

                    text = getattr(resp, "text", None)
                    if text:
                        return text

                    # empty response: work out why rather than returning None
                    reason = "unknown"
                    try:
                        cand = resp.candidates[0]
                        reason = str(getattr(cand, "finish_reason", "unknown"))
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"{name} returned no text (finish_reason={reason}). "
                        f"If this says MAX_TOKENS, raise max_tokens.")
                gm = gclient.GenerativeModel(name)
                resp = gm.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": max_tokens,
                                       "temperature": temperature},
                )
                return resp.text

        except Exception as e:
            last_err = e
            msg = str(e)
            if "PerDay" in msg or "RequestsPerDayPerProject" in msg:
                raise RuntimeError(
                    f"{model} ({name}) hit its DAILY free-tier limit.\n"
                    f"Options: switch model with "
                    f"llm.set_model('gemini', 'gemini-2.5-flash-lite'), "
                    f"wait until the quota resets, or enable billing.\n"
                    f"Original: {msg[:200]}")
            if "limit: 0" in msg:
                # no free-tier allocation for this model at all, so retrying
                # will never help
                raise RuntimeError(
                    f"{model} ({name}) has no quota on this key.\n"
                    f"Run llm.list_gemini_models() to see what is available, "
                    f"then llm.set_model('gemini', '<name>').\n"
                    f"Original error: {msg[:200]}")
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))

    raise RuntimeError(f"{model} failed after {retries} attempts: {last_err}")


def available_models() -> list:
    """Which models we actually have keys for."""
    have = []
    for key, spec in MODELS.items():
        env = {"openai": "OPENAI_API_KEY",
               "anthropic": "ANTHROPIC_API_KEY",
               "google": "GOOGLE_API_KEY"}[spec["provider"]]
        if os.environ.get(env):
            have.append(key)
    return have


# Model availability shifts: names get deprecated for new users, free-tier
# quotas move. Rather than guessing, try candidates until one answers.
GEMINI_CANDIDATES = [
    "gemini-flash-lite-latest",   # alias, tracks whatever is current
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]


def find_working_model(candidates: list = None, verbose: bool = True) -> str:
    """
    Try each candidate with a tiny prompt and return the first that works.

    Sets it as the active gemini model. Costs one request per candidate
    tried, and stops at the first success.
    """
    candidates = candidates or GEMINI_CANDIDATES
    original = MODELS["gemini"]["name"]

    for name in candidates:
        MODELS["gemini"]["name"] = name
        _clients.pop("google", None)
        try:
            out = call_model("Reply with the single word OK.",
                             model="gemini", max_tokens=200, retries=1)
            if out and out.strip():
                if verbose:
                    print(f"working: {name}")
                return name
        except Exception as e:
            msg = str(e)
            if "NOT_FOUND" in msg or "no longer available" in msg:
                reason = "not available"
            elif "PerDay" in msg or "limit: 0" in msg:
                reason = "quota exhausted"
            elif "429" in msg:
                reason = "rate limited"
            else:
                reason = msg[:60]
            if verbose:
                print(f"  {name:26s} {reason}")

    MODELS["gemini"]["name"] = original
    _clients.pop("google", None)
    raise RuntimeError(
        "no working gemini model found. Either every candidate is out of "
        "quota for today, or the key has no free-tier allocation. Options: "
        "wait for the daily reset, create the key under a new Google Cloud "
        "project, or enable billing.")
