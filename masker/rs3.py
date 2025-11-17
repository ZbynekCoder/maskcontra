from typing import List, Dict, Any, Optional


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def find_segments(rs3_text: str) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(rs3_text)
    body = root.find("body")
    seg_elems = (body.findall("segment") if body is not None else root.findall(".//segment"))
    segments: List[Dict[str, Any]] = []
    for seg in seg_elems:
        sid = seg.get("id")
        try:
            seg_id_val = int(sid) if sid is not None else None
        except Exception:
            seg_id_val = sid
        text = seg.text or ""
        segments.append({"seg_id": seg_id_val, "text": text})
    return segments


def crop_window_with_closure(
    rs3_text: str,
    center_seg_id: int,
    window_d: int,
    mask_token: str = "[MASK]",
    include_center_siblings: bool = True,
    sibling_window_d: Optional[int] = None,
) -> str:
    """
    Window union + ancestor group closure around the center segment.
    - Keep segments within [cpos-d, cpos+d].
    - If include_center_siblings: also expand one window around siblings (segments and group descendants).
    - Mask only the center segment text with mask_token.
    - Keep only used relation types in header.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(rs3_text)
    header = root.find("header")
    body = root.find("body")

    seg_elems = (body.findall("segment") if (body is not None) else root.findall(".//segment"))
    group_elems = (body.findall("group") if (body is not None) else root.findall(".//group"))
    n = len(seg_elems)
    if n == 0:
        return rs3_text

    id_order = [s.get("id") for s in seg_elems]
    sid_to_pos = {(sid or ""): i for i, sid in enumerate(id_order, start=1) if sid is not None}
    seg_by_id = {(s.get("id") or ""): s for s in seg_elems if s.get("id") is not None}
    group_by_id = {(g.get("id") or ""): g for g in group_elems if g.get("id") is not None}

    children_segments: Dict[str, List[str]] = {}
    children_groups: Dict[str, List[str]] = {}

    def _push(dic: Dict[str, List[str]], key: Optional[str], val: Optional[str]) -> None:
        if key is None or val is None:
            return
        k, v = str(key), str(val)
        dic.setdefault(k, []).append(v)

    for s in seg_elems:
        _push(children_segments, s.get("parent"), s.get("id"))
    for g in group_elems:
        _push(children_groups, g.get("parent"), g.get("id"))

    center_sid = str(center_seg_id)
    cpos = sid_to_pos.get(center_sid)
    if cpos is None or cpos < 1 or cpos > n:
        # If the id is not in sequence, keep original text
        return rs3_text

    d_center = int(window_d)
    d_sib = int(sibling_window_d) if (sibling_window_d is not None) else d_center

    keep_seg_ids: set[str] = set()

    def _add_window(pos_center: int, d: int) -> None:
        start_pos = max(1, pos_center - d)
        end_pos = min(n, pos_center + d)
        for p in range(start_pos, end_pos + 1):
            sid = id_order[p - 1]
            if sid:
                keep_seg_ids.add(str(sid))

    # 1) center window
    _add_window(cpos, d_center)

    # 2) expand around siblings
    if include_center_siblings:
        parent_id = seg_by_id.get(center_sid).get("parent") if center_sid in seg_by_id else None
        if parent_id is not None:
            parent_id = str(parent_id)
            # siblings segments
            sib_segs = [sid for sid in children_segments.get(parent_id, []) if sid != center_sid]
            for sid in sib_segs:
                pos = sid_to_pos.get(sid)
                if pos is not None:
                    _add_window(pos, d_sib)

            # group descendants
            def _descendant_seg_ids_of_group(gid: str) -> List[str]:
                out: List[str] = []
                out.extend(children_segments.get(gid, []))
                for child_gid in children_groups.get(gid, []):
                    out.extend(_descendant_seg_ids_of_group(child_gid))
                return out

            sib_groups = children_groups.get(parent_id, [])
            for gid in sib_groups:
                for sid in _descendant_seg_ids_of_group(gid):
                    pos = sid_to_pos.get(sid)
                    if pos is not None:
                        _add_window(pos, d_sib)

    # 3) ancestor closure for groups
    keep_group_ids: set[str] = set()

    def _ascend_group_chain_from_seg(sid: str) -> None:
        node = seg_by_id.get(sid)
        cur_parent = node.get("parent") if node is not None else None
        while cur_parent:
            cur_parent = str(cur_parent)
            if cur_parent in keep_group_ids:
                g = group_by_id.get(cur_parent)
                cur_parent = g.get("parent") if g is not None else None
                continue
            keep_group_ids.add(cur_parent)
            g = group_by_id.get(cur_parent)
            cur_parent = g.get("parent") if g is not None else None

    for sid in list(keep_seg_ids):
        _ascend_group_chain_from_seg(sid)

    # 4) filter header relation types
    used_relnames: set[str] = set()
    for sid in keep_seg_ids:
        s = seg_by_id.get(sid)
        rn = s.get("relname") if s is not None else None
        if rn:
            used_relnames.add(str(rn))
    for gid in keep_group_ids:
        g = group_by_id.get(gid)
        rn = g.get("relname") if g is not None else None
        if rn:
            used_relnames.add(str(rn))

    # 5) rebuild RS3
    import xml.etree.ElementTree as ET
    new_root = ET.Element(root.tag, root.attrib)

    if header is not None:
        new_header = ET.fromstring(ET.tostring(header, encoding="unicode"))
        rels = new_header.find("relations") or new_header.find("reltypes")
        if rels is not None:
            new_rels = ET.Element(rels.tag)
            for r in rels.findall("rel"):
                name = r.get("name")
                if name and (not used_relnames or name in used_relnames):
                    new_rels.append(ET.fromstring(ET.tostring(r, encoding="unicode")))
            new_header.remove(rels)
            new_header.append(new_rels)
        new_root.append(new_header)

    new_body = ET.Element("body")
    for child in list(body) if body is not None else []:
        if child.tag == "segment":
            sid = child.get("id")
            if sid and str(sid) in keep_seg_ids:
                new_seg = ET.Element("segment", attrib=dict(child.attrib))
                new_seg.text = (mask_token if str(sid) == center_sid else (child.text or ""))
                new_body.append(new_seg)
        elif child.tag == "group":
            gid = child.get("id")
            if gid and str(gid) in keep_group_ids:
                new_group = ET.Element("group", attrib=dict(child.attrib))
                new_body.append(new_group)
    new_root.append(new_body)

    return ET.tostring(new_root, encoding="unicode")
