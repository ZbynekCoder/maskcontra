import json
from typing import Optional, Dict, List

from openai import OpenAI
import config
from . import prompts


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class LLM:
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
        self.req_timeout = config.LLM_TIMEOUT

    def _chat(self, messages: List[Dict[str, str]], temperature: float) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
            timeout=self.req_timeout,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("empty LLM content")
        return content

    def fill_mask(self, context_nl: str) -> str:
        msgs = prompts.build_fill_mask_messages(context_nl)
        text = self._chat(msgs, temperature=max(0.0, min(1.0, self.temperature)))
        # Strip common wrappers
        text = text.strip().strip('"').strip("'").strip("`")
        text = text.replace("[MASK]", "").strip()
        if not text:
            raise RuntimeError("empty completion after cleanup")
        return text

    def classify_nli(self, premise: str, hypothesis: str) -> Dict[str, Optional[str]]:
        msgs = prompts.build_nli_messages(premise, hypothesis)
        raw = self._chat(msgs, temperature=0.0).strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
        obj = json.loads(raw)
        label = str(obj.get("label", "")).strip().lower()
        conf = obj.get("confidence", None)
        if conf is None:
            raise RuntimeError("NLI response missing 'confidence'")
        conf = _clip01(float(conf))
        return {"label": label, "confidence": conf}
