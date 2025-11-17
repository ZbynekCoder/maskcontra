# Self-Contradiction Detection Project Configuration

import os
import dotenv

dotenv.load_dotenv()


# ============ LLM API ============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)  # None -> provider default
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
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
