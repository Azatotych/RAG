from huggingface_hub import snapshot_download
from pathlib import Path

BASE_DIR = Path(r"E:\projects\rag")
MODELS_DIR = BASE_DIR / "models"

MODELS = [
    # Быстрая модель для baseline
    ("sergeyzh/rubert-mini-sts", "rubert-mini-sts"),
    # Тяжёлая, более точная LaBSE-ru-sts
    ("sergeyzh/LaBSE-ru-sts", "LaBSE-ru-sts"),
]

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for hf_id, local_name in MODELS:
        target_dir = MODELS_DIR / local_name
        print(f"Скачиваю {hf_id} -> {target_dir}")
        snapshot_download(
            repo_id=hf_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Готово: {target_dir}")

if __name__ == "__main__":
    main()
