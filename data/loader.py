import json
from typing import List, Dict, Any


def _normalize_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Fail fast: require text
    if "text" not in obj or obj["text"] is None:
        raise ValueError("record missing 'text'")
    text = obj["text"]
    iid = obj.get("unique id") or obj.get("id")
    label = obj.get("label", "unknown")
    return {"id": iid, "text": text, "label": label, "meta": obj}


def load(path: str) -> List[Dict[str, Any]]:
    """
    Supported inputs:
      - JSONL: one JSON per line
      - JSON array: [{...}, ...]
      - JSON object:
          * single record with text
          * or {"pos": {...}, "neg": {...}} where leaves are dicts with text
    Returns list of {"id", "text", "label", "meta"}.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []

    # Try parse as JSON first; if fails, treat as JSONL
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        items: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(_normalize_record(obj))
        return items

    # pos/neg map
    if isinstance(data, dict) and ("pos" in data or "neg" in data):
        result: List[Dict[str, Any]] = []
        for lb in ("pos", "neg"):
            block = data.get(lb, {})
            if not isinstance(block, dict):
                continue
            for _, v in block.items():
                if not isinstance(v, dict):
                    continue
                rec = _normalize_record(v)
                result.append({"id": rec["id"], "text": rec["text"], "label": lb, "meta": rec["meta"]})
        return result

    # array
    if isinstance(data, list):
        return [_normalize_record(obj) for obj in data if isinstance(obj, dict)]

    # single object
    if isinstance(data, dict):
        rec = _normalize_record(data)
        return [rec]

    raise ValueError("unsupported dataset format")
