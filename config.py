# Self-Contradiction Detection Project Configuration

import os
import dotenv

dotenv.load_dotenv()


# ============ LLM API ============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)  # None -> provider default
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# ============ NLI / Threshold ============
NLI_CONFIDENCE_THRESHOLD = float(os.getenv("NLI_CONFIDENCE_THRESHOLD", "0.7"))

# ============ HF RST Parser ============
HF_MODEL_NAME = "tchewik/isanlp_rst_v3"
HF_MODEL_VERSION = os.getenv("HF_MODEL_VERSION", "gumrrg")
CUDA_DEVICE = int(os.getenv("CUDA_DEVICE", "-1"))

# ============ Concurrency / Context ============
CONCURRENCY = int(os.getenv("CONCURRENCY", "16"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "15"))
CONTEXT_INCLUDE_CENTER_SIBLINGS = True
CONTEXT_WINDOW_D_SIBLING = int(os.getenv("CONTEXT_WINDOW_D_SIBLING", "10"))

# ============ Output ============
OUT_DIR = os.getenv("OUT_DIR", "out")
SAVE_MASKED = os.getenv("SAVE_MASKED", "true").lower() in ("1", "true", "yes")

# ============ Logging (unused here but reserved) ============
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "contradiction_check.log")

# ============ Local T5 Biasing ============
FILLER_MODEL_NAME = os.getenv("FILLER_MODEL_NAME", "t5-base")
FILLER_NUM_BEAMS = int(os.getenv("FILLER_NUM_BEAMS", "6"))
FILLER_MAX_NEW_TOKENS = int(os.getenv("FILLER_MAX_NEW_TOKENS", "64"))
# 生成策略：similar | opposite
FILLER_MODE = os.getenv("FILLER_MODE", "similar").lower()
# 词表级偏置强度
FILLER_COPY_ALPHA = 1.0     # 原文本子词拷贝/反拷贝
FILLER_EMB_GAMMA = 0.6       # 嵌入亲和度权重
FILLER_NEGATION_DELTA = 0.4  # 否定词极性

FILLER_MIN_TOKENS = 1           # 最少生成 3 个 token
FILLER_NO_REPEAT_NGRAM = 3 # ngram 去重
FILLER_REPETITION_PENALTY = 1.05
FILLER_LENGTH_PENALTY = 1.02
FILLER_PUNCT_STEPS = 2

# 多候选 + 采样
FILLER_CANDIDATES = int(os.getenv("FILLER_CANDIDATES", "4"))
FILLER_SAMPLING = os.getenv("FILLER_SAMPLING", "true").lower() in ("1", "true", "yes")
FILLER_TOP_P = float(os.getenv("FILLER_TOP_P", "0.9"))
FILLER_TOP_K = int(os.getenv("FILLER_TOP_K", "50"))
FILLER_TEMPERATURE = float(os.getenv("FILLER_TEMPERATURE", "0.8"))


NEGATION_WORDS = os.getenv(
    "NEGATION_WORDS",
    "not,no,never,none,nobody,nothing,nowhere,neither,nor,without,n't,"
    "cannot,can't,won't,don't,doesn't,didn't,isn't,aren't,wasn't,weren't,"
    "shouldn't,couldn't,wouldn't,mustn't,ain't"
).split(",")

FILLER_BAN_WORDS = [w for w in os.getenv("FILLER_BAN_WORDS", "").split(",") if w.strip()]