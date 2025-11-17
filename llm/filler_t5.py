import re
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import (
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessor,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
)

import config


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype_for(device: str):
    return torch.float16 if device == "cuda" else torch.float32


def _flatten_ids(nested_ids) -> List[int]:
    if not isinstance(nested_ids, list):
        return []
    if len(nested_ids) > 0 and isinstance(nested_ids[0], int):
        return list(set(int(t) for t in nested_ids))
    out = []
    for arr in nested_ids:
        if isinstance(arr, list):
            out.extend(arr)
        elif isinstance(arr, int):
            out.append(arr)
    return list(set(out))


class BanTokenUntilStep(LogitsProcessor):
    """在前 min_steps 步禁止某个 token（这里用来延迟 <extra_id_1>）。"""

    def __init__(self, token_id: int, min_steps: int):
        self.token_id = int(token_id)
        self.min_steps = int(min_steps)

    def __call__(self, input_ids, scores):
        # 对于 T5，decoder 的第一个步骤通常只有一个起始 token（decoder_start_token）
        cur_len = input_ids.shape[1]
        if cur_len < (1 + self.min_steps):
            scores[:, self.token_id] = -1e9
        return scores


class BanFirstStepPunct(LogitsProcessor):
    """第一步禁止标点起步，避免一上来就是 '.' ',' 等。"""

    def __init__(self, tok, punct_list=None):
        punct_list = punct_list or [".", ",", ";", ":", "!", "?", "—", "-", "..."]
        # 将标点转成 id（兼容 SentencePiece）
        ids = tok(punct_list, add_special_tokens=False).input_ids
        self.punct_ids = list({tid for arr in ids for tid in (arr if isinstance(arr, list) else [arr])})

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[1]
        if cur_len == 1 and self.punct_ids:
            scores[:, self.punct_ids] = -1e9
        return scores


class CopyLexiconLogitsProcessor(LogitsProcessor):
    # 对来自 ref_text 的子词 ids 做加分（similar）或减分（opposite）
    def __init__(self, copy_ids: List[int], alpha: float):
        self.copy_ids = list(set(int(x) for x in copy_ids if x is not None))
        self.alpha = float(alpha)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.copy_ids and self.alpha != 0.0:
            scores[:, self.copy_ids] += self.alpha
        return scores


class NegationPolarityLogitsProcessor(LogitsProcessor):
    # 对否定词做极性加权：similar 减分；opposite 加分
    def __init__(self, tok, neg_words: List[str], delta: float, mode: str):
        ids = tok(neg_words or [], add_special_tokens=False).input_ids
        self.neg_ids = _flatten_ids(ids)
        self.delta = float(delta) * (1.0 if mode == "opposite" else -1.0)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.neg_ids and self.delta != 0.0:
            scores[:, self.neg_ids] += self.delta
        return scores


class EmbeddingAffinityLogitsProcessor(LogitsProcessor):
    # 使用 ref_text 子词嵌入的质心与词表嵌入的余弦相似度作偏置
    def __init__(self, emb_weight: torch.Tensor, ref_ids: List[int], gamma: float, mode: str):
        if not ref_ids:
            self.bias = None
            self.gamma = 0.0
            self.sign = 1.0
            return
        with torch.no_grad():
            ref_ids_t = torch.tensor(ref_ids, dtype=torch.long, device=emb_weight.device)
            ref_vecs = emb_weight[ref_ids_t]
            mask = (ref_vecs.abs().sum(dim=1) > 0)
            ref_vecs = ref_vecs[mask] if mask.any() else ref_vecs
            if ref_vecs.numel() == 0:
                self.bias = None
                self.gamma = 0.0
                self.sign = 1.0
                return
            centroid = ref_vecs.mean(dim=0, keepdim=True)
            w = F.normalize(emb_weight, dim=1)
            c = F.normalize(centroid, dim=1)
            sim = (w @ c.t()).squeeze(1)  # [-1,1]
            self.bias = sim
        self.gamma = float(abs(gamma))
        self.sign = 1.0 if mode == "similar" else -1.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.bias is None or self.gamma == 0.0:
            return scores
        return scores + self.sign * self.gamma * self.bias.unsqueeze(0)


class Filler:
    """
    本地 T5 填空器：只实现 fill_mask(context_nl, ref_text, mode)
    - [MASK] -> <extra_id_0>；取 <extra_id_0> 与 <extra_id_1> 之间作为答案
    - 通过 LogitsProcessor 在词表级注入“相似/相反”偏置
    """

    def __init__(self):
        self.device = _device()
        self.model_name = config.FILLER_MODEL_NAME
        self.tok = T5Tokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, torch_dtype=_dtype_for(self.device)
        ).to(self.device).eval()

        self.num_beams = int(config.FILLER_NUM_BEAMS)
        self.max_new_tokens = int(config.FILLER_MAX_NEW_TOKENS)

        self.emb_weight: torch.Tensor = self.model.get_input_embeddings().weight.detach().to(self.device)

        self.neg_words = [w.strip() for w in config.NEGATION_WORDS if w.strip()]
        self.ban_words = [w.strip() for w in config.FILLER_BAN_WORDS if w.strip()]
        self.copy_alpha = float(config.FILLER_COPY_ALPHA)
        self.emb_gamma = float(config.FILLER_EMB_GAMMA)
        self.neg_delta = float(config.FILLER_NEGATION_DELTA)

        self.min_tokens = int(getattr(config, "FILLER_MIN_TOKENS", 3))
        self.no_repeat_ngram = int(getattr(config, "FILLER_NO_REPEAT_NGRAM", 3))
        self.rep_penalty = float(getattr(config, "FILLER_REPETITION_PENALTY", 1.08))
        self.len_penalty = float(getattr(config, "FILLER_LENGTH_PENALTY", 1.05))
        self.ban_punct_first = bool(getattr(config, "FILLER_BAN_PUNCT_FIRST", True))

    def _dyn_min_tokens(self, ref_text: str) -> int:
        if not ref_text:
            return max(1, int(getattr(config, "FILLER_MIN_TOKENS", 1)))
        ids = self.tok([ref_text], add_special_tokens=False).input_ids[0]
        L = max(1, len(ids))
        # 短句允许 1–2 token，长句给 3–5 token
        if L <= 6:
            return 1
        if L <= 16:
            return 2
        return 3

    @torch.no_grad()
    def fill_mask_candidates(self, context_nl: str, ref_text: Optional[str], mode: str, k: int) -> List[str]:
        mode = (mode or "similar").lower()
        inp = context_nl.replace("[MASK]", "<extra_id_0>")
        enc = self.tok(inp, return_tensors="pt").to(self.device)

        processors = LogitsProcessorList()
        # 软偏置：copy / embedding / negation（同你已有逻辑，使用较小权重）
        ref_ids = _flatten_ids(self.tok([ref_text], add_special_tokens=False).input_ids) if ref_text else []
        alpha = self.copy_alpha if mode == "similar" else -self.copy_alpha
        if ref_ids and alpha != 0.0:
            processors.append(CopyLexiconLogitsProcessor(ref_ids, alpha))
        if ref_ids and self.emb_gamma != 0.0:
            processors.append(EmbeddingAffinityLogitsProcessor(self.emb_weight, ref_ids, self.emb_gamma, mode))
        if self.neg_words and self.neg_delta != 0.0:
            processors.append(NegationPolarityLogitsProcessor(self.tok, self.neg_words, self.neg_delta, mode))
        # 首步弱惩罚标点
        processors.append(
            PunctPenaltyLogitsProcessor(self.tok, steps=getattr(config, "FILLER_PUNCT_STEPS", 2), penalty=1.2))

        min_tokens = self._dyn_min_tokens(ref_text or "")
        do_sample = bool(getattr(config, "FILLER_SAMPLING", True))
        num_return_sequences = max(1, int(k))
        gen = self.model.generate(
            **enc,
            num_beams=1 if do_sample else 5,
            do_sample=do_sample,
            top_p=getattr(config, "FILLER_TOP_P", 0.9) if do_sample else None,
            top_k=getattr(config, "FILLER_TOP_K", 50) if do_sample else None,
            temperature=getattr(config, "FILLER_TEMPERATURE", 0.8) if do_sample else None,
            max_new_tokens=max(1, self.max_new_tokens),
            min_new_tokens=max(1, min_tokens),
            early_stopping=True,
            no_repeat_ngram_size=max(0, self.no_repeat_ngram),
            repetition_penalty=max(1.0, self.rep_penalty),
            length_penalty=max(0.0, self.len_penalty),
            logits_processor=processors if len(processors) > 0 else None,
            num_return_sequences=num_return_sequences,
        )
        outs = []
        for seq in gen:
            dec = self.tok.decode(seq, skip_special_tokens=False)
            m = re.search(r"<extra_id_0>(.*?)<extra_id_1>", dec, flags=re.S)
            ans = (m.group(1) if m else dec).strip()
            ans = ans.replace("<extra_id_0>", "").replace("<extra_id_1>", "")
            ans = ans.replace("<pad>", "").replace("</s>", "").strip()
            if ans:
                outs.append(ans)
        # 去重保序
        seen, uniq = set(), []
        for s in outs:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq or ["."]

    @torch.no_grad()
    def fill_mask(self, context_nl: str, ref_text: Optional[str], mode: str) -> str:
        return self.fill_mask_candidates(context_nl, ref_text, mode, k=1)[0]


class PunctPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, tok, steps=2, penalty=1.2):
        ids = tok([".", ",", ";", ":", "!", "?", "—", "-", "..."], add_special_tokens=False).input_ids
        self.punct_ids = list({tid for arr in ids for tid in (arr if isinstance(arr, list) else [arr])})
        self.steps = int(steps)
        self.penalty = float(penalty)

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[1]
        if cur_len <= self.steps and self.punct_ids:
            scores[:, self.punct_ids] -= self.penalty
        return scores
