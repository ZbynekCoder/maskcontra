from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


def extract_segments_dict(rs3_text: str, trim: bool = True) -> Dict[str, str]:
    if not rs3_text:
        return {}
    root = ET.fromstring(rs3_text)
    body = root.find("body")
    seg_elems = (body.findall("segment") if body is not None else root.findall(".//segment"))
    result: Dict[str, str] = {}
    for seg in seg_elems:
        sid = seg.get("id")
        if sid is None:
            continue
        text = seg.text or ""
        result[str(sid)] = text.strip() if trim else text
    return result


def to_masked_nl(segments: Dict[str, str]) -> str:
    # Concatenate ordered segments with "..." bridging gaps and edges.
    numeric: List[Tuple[int, str]] = []
    for k, v in segments.items():
        numeric.append((int(str(k)), (v or "").strip()))
    numeric.sort(key=lambda x: x[0])

    ctx: List[str] = []
    prev_id: Optional[int] = None
    ctx.append("...")
    for cur_id, txt in numeric:
        if prev_id is not None and (cur_id - prev_id) > 1:
            if not ctx[-1].endswith("..."):
                ctx.append("...")
        ctx.append(txt)
        prev_id = cur_id
    ctx.append("...")
    return " ".join(ctx)
