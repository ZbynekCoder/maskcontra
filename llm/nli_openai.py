import json
from typing import Optional, Dict

from openai import OpenAI

import config
from .prompts import build_nli_messages


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class NLI:
    """
    Simple chat-completion LLM wrapper with two methods:
      - fill_mask(context_nl) -> str
      - classify_nli(premise, hypothesis) -> {"label": str, "confidence": float}
    Fail fast on network/parse errors.
    """

    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_API_BASE)
        self.model = config.OPENAI_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.timeout = config.LLM_TIMEOUT

    def classify_nli(self, premise: str, hypothesis: str) -> Dict[str, Optional[str]]:
        msgs = build_nli_messages(premise, hypothesis)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=config.LLM_TEMPERATURE,
            messages=msgs,
            timeout=self.timeout,
        )
        raw = (resp.choices[0].message.content or "").strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
        obj = json.loads(raw)
        label = str(obj.get("label", "")).strip().lower()
        conf = _clip01(float(obj.get("confidence", 0.0)))
        return {"label": label, "confidence": conf}
