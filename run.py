from datetime import datetime
import os

import config
from pipeline import run

if __name__ == "__main__":
    dataset_path = os.getenv("DATASET_PATH", "CONTRADOC/ContraDoc.json")
    out_dir = f"{config.OUT_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    res = run(
        dataset_path=dataset_path,
        out_dir=out_dir,
        hf_version=config.HF_MODEL_VERSION,
        cuda_device=config.CUDA_DEVICE,
        save_masked=config.SAVE_MASKED,
        lower=0,
        upper=1,
    )
    print(f"[DONE] Summary: {res['results_path']}  Count: {res['count']}")
