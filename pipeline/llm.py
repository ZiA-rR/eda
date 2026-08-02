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

import os
import time
from typing import Optional

# Model names per provider. Update these as versions change.
MODELS = {
    "gpt":    {"provider": "openai",    "name": "gpt-4o"},
    "claude": {"provider": "anthropic", "name": "claude-sonnet-4-5"},
    "gemini": {"provider": "google",    "name": "gemini-2.0-flash"},
}

# The three used for scoring. Deliberately one from each family.
SCORING_MODELS = ["gpt", "claude", "gemini"]

_clients = {}


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
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        _clients[provider] = genai
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
                gm = client.GenerativeModel(name)
                resp = gm.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": max_tokens,
                                       "temperature": temperature},
                )
                return resp.text

        except Exception as e:
            last_err = e
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
