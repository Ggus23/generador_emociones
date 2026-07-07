from pathlib import Path
CLASES = [
    "angry",
    "neutral",
    "sad",
    "surprise",
    "bored"
]
RUTA_BASE = Path("optencion_dataset")
def main():
    for clase in CLASES:
        (RUTA_BASE / "capturas" / clase).mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for clase in CLASES:
            (RUTA_BASE / "final" / split / clase).mkdir(parents=True, exist_ok=True)
print("Estructura creada correctamente.")
if __name__ == "__main__":
    main()