from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

def get_encoder(name: str = "mini") -> SentenceTransformer:
    """
    name: 'mini'  -> sergeyzh/rubert-mini-sts
          'labse' -> sergeyzh/LaBSE-ru-sts
    """
    if name == "mini":
        local_dir = MODELS_DIR / "rubert-mini-sts"
        hf_id = "sergeyzh/rubert-mini-sts"
    elif name == "labse":
        local_dir = MODELS_DIR / "LaBSE-ru-sts"
        hf_id = "sergeyzh/LaBSE-ru-sts"
    else:
        raise ValueError(f"Unknown encoder: {name}")

    if local_dir.exists():
        return SentenceTransformer(str(local_dir))
    else:
        # на всякий случай: если папки нет, скачает по имени с HF
        return SentenceTransformer(hf_id)
