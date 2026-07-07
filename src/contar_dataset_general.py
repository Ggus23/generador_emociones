from pathlib import Path
RUTA = Path("dataset_emociones_general/final")
for split in ["train", "val", "test"]:
    print(f"\n--- {split.upper()} ---")
    for clase in sorted((RUTA / split).iterdir()):
        if clase.is_dir():
            total = len([x for x in clase.iterdir() if x.is_file()])
            print(f"{clase.name}: {total}")