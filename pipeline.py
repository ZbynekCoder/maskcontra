import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

import config
from data.loader import load as load_dataset
from rst_parser import RSTParser
from masker.rs3 import read_text, write_text, find_segments, crop_window_with_closure
from masker.context import extract_segments_dict, to_masked_nl
from llm import Filler, NLI


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_id(raw: str) -> str:
    s = "".join(
        (ch if ("a" <= ch <= "z" or "A" <= ch <= "Z" or "0" <= ch <= "9" or ch in ("_", "-")) else "_") for ch in
        str(raw))
    s = s.strip("_")
    return s or "sample"


def evaluate_one_rs3(filler: Filler, nli: NLI, rs3_text: str) -> Dict[str, Any]:
    segs = find_segments(rs3_text)
    thr = config.NLI_CONFIDENCE_THRESHOLD

    tasks: List[Any] = []
    for meta in segs:
        orig = meta["text"]
        sid = meta["seg_id"]

        # Precompute masked RS3 per segment (centered window for fill-mask context)
        masked_rs3 = crop_window_with_closure(
            rs3_text,
            center_seg_id=sid,
            window_d=config.CONTEXT_WINDOW,
            mask_token="[MASK]",
            include_center_siblings=config.CONTEXT_INCLUDE_CENTER_SIBLINGS,
            sibling_window_d=config.CONTEXT_WINDOW_D_SIBLING,
        )
        tasks.append((sid, orig, masked_rs3))

    results_ordered: Dict[int, Dict[str, Any]] = {}

    def worker(payload):
        sid, orig, masked_rs3 = payload
        context_nl = to_masked_nl(extract_segments_dict(masked_rs3))
        # 生成多候选
        cands = filler.fill_mask_candidates(context_nl, ref_text=orig, mode=config.FILLER_MODE,
                                            k=getattr(config, "FILLER_CANDIDATES", 4))

        # NLI 评估并挑选最佳：优先 非矛盾(= entail/neutral) 且置信最高；若全是 contradiction，选置信度最低的矛盾（最不确定）
        best = None
        best_score = -1.0
        fallback = None
        fallback_score = 1e9
        for s in cands:
            v = nli.classify_nli(premise=orig, hypothesis=s)
            label, conf = v["label"], float(v["confidence"])
            if label != "contradiction":
                score = conf  # 越大越好
                if score > best_score:
                    best, best_score = (s, v), score
            else:
                # 作为兜底：选置信度最低的 contradiction
                if conf < fallback_score:
                    fallback, fallback_score = (s, v), conf

        if best is None:
            llm_text, verdict = fallback
        else:
            llm_text, verdict = best

        label, conf = verdict["label"], verdict["confidence"]
        contradictory = (label == "contradiction" and conf >= thr)
        return sid, {
            "segment_id": sid,
            "orig_text": orig,
            "llm_text": llm_text,
            "nli_label": label,
            "nli_confidence": conf,
            "contradictory": contradictory,
        }

    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, config.CONCURRENCY)) as ex:
            futures = [ex.submit(worker, t) for t in tasks]
            for fut in as_completed(futures):
                sid, item = fut.result()
                results_ordered[sid] = item

    results: List[Dict[str, Any]] = []
    contradict_count = 0
    for meta in segs:
        sid = meta["seg_id"]
        item = results_ordered[sid]
        results.append(item)
        if item["contradictory"]:
            contradict_count += 1

    return {
        "segments": results,
        "stats": {"total": len(results), "contradict_count": contradict_count},
    }


def run(
        dataset_path: str,
        out_dir: str,
        hf_version: str,
        cuda_device: int,
        save_masked: bool,
        lower: Optional[int] = None,
        upper: Optional[int] = None,
) -> Dict[str, Any]:
    out_rs3_dir = os.path.join(out_dir, "rs3")
    out_eval_dir = os.path.join(out_dir, "eval")
    out_masked_dir = os.path.join(out_dir, "masked")
    ensure_dir(out_rs3_dir)
    ensure_dir(out_eval_dir)
    if save_masked:
        ensure_dir(out_masked_dir)

    # 1) Load dataset
    items = load_dataset(dataset_path)
    if lower is not None and upper is not None:
        items = items[lower:upper]
    elif lower is not None:
        items = items[lower:]
    elif upper is not None:
        items = items[:upper]

    # 2) Init services
    rst = RSTParser(hf_model_name=config.HF_MODEL_NAME, hf_version=hf_version, cuda_device=cuda_device)
    filler = Filler()
    nli = NLI()

    # 3) Parse -> RS3
    for it in items:
        iid = sanitize_id(it["id"])
        text = it["text"]
        rs3_path = os.path.join(out_rs3_dir, f"{iid}.rs3")
        rst.save_rs3(text, rs3_path)


    # 4) Evaluate
    results_path = os.path.join(out_eval_dir, "results.jsonl")
    with open(results_path, "w", encoding="utf-8") as fout:
        pass  # truncate

    processed = 0
    for it in items:
        raw_id = it["id"]
        label = it.get("label", "unknown")
        meta = it.get("meta", {})
        iid = sanitize_id(raw_id)
        rs3_path = os.path.join(out_rs3_dir, f"{iid}.rs3")
        rs3_text = read_text(rs3_path)

        if save_masked:
            # Save per-segment windows with mask for debugging
            segs = find_segments(rs3_text)
            for idx, meta_seg in enumerate(segs, start=1):
                masked_txt = crop_window_with_closure(
                    rs3_text,
                    center_seg_id=meta_seg["seg_id"],
                    window_d=config.CONTEXT_WINDOW,
                    mask_token="[MASK]",
                    include_center_siblings=config.CONTEXT_INCLUDE_CENTER_SIBLINGS,
                    sibling_window_d=config.CONTEXT_WINDOW_D_SIBLING,
                )
                write_text(os.path.join(out_masked_dir, f"{iid}_seg{idx}.rs3"), masked_txt)

        eval_res = evaluate_one_rs3(filler, nli, rs3_text)

        per_id_json = {"id": raw_id, "label": label, "meta": meta, "eval": eval_res}
        per_id_path = os.path.join(out_eval_dir, f"{iid}.json")
        with open(per_id_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(per_id_json, ensure_ascii=False, indent=2))

        summary = {
            "id": raw_id,
            "label": label,
            "total": eval_res["stats"]["total"],
            "contradict_count": eval_res["stats"]["contradict_count"],
            "contradict_segments": [
                {
                    "segment_id": seg["segment_id"],
                    "orig_text": seg["orig_text"],
                    "llm_text": seg["llm_text"],
                    "nli_label": seg["nli_label"],
                    "nli_confidence": seg["nli_confidence"],
                }
                for seg in eval_res["segments"]
                if seg["contradictory"]
            ],
        }
        with open(results_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

        processed += 1

    return {"results_path": results_path, "count": processed}
