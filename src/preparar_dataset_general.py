import cv2
import random
import shutil
from pathlib import Path
random.seed(42)
CLASES = [
    "angry",
    "neutral",
    "sad",
    "surprise",
    "bored"
]
RUTA_CAPTURAS = Path("dataset_emociones_general/capturas")
RUTA_FINAL = Path("dataset_emociones_general/final")
PORC_TRAIN = 0.70
PORC_VAL = 0.15
PORC_TEST = 0.15
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
def limpiar_final():
    if RUTA_FINAL.exists():
        shutil.rmtree(RUTA_FINAL)
    for split in ["train", "val", "test"]:
        for clase in CLASES:
            (RUTA_FINAL / split / clase).mkdir(parents=True, exist_ok=True)

def obtener_imagenes(ruta):
    return [
    p for p in ruta.iterdir()
    if p.is_file() and p.suffix.lower() in EXTENSIONES
    ]
def recortar_rostro(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = CASCADE.detectMultiScale(
    gris,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(80, 80)
    )
    if len(rostros) == 0:
        return None
    rostros = sorted(rostros, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = rostros[0]
    margen = int(0.25 * max(w, h))
    x1 = max(0, x - margen)
    y1 = max(0, y - margen)
    x2 = min(imagen.shape[1], x + w + margen)
    y2 = min(imagen.shape[0], y + h + margen)
    rostro = imagen[y1:y2, x1:x2]
    if rostro.size == 0:
        return None
    rostro = cv2.resize(rostro, (160, 160))
    return rostro
def guardar_recorte(origen, destino):
    imagen = cv2.imread(str(origen))
    if imagen is None:
        return False
    rostro = recortar_rostro(imagen)
    if rostro is None:
        return False
    cv2.imwrite(str(destino), rostro)
    return True
def main():
    limpiar_final()
    for clase in CLASES:
        ruta_clase = RUTA_CAPTURAS / clase
        if not ruta_clase.exists():
            print(f"No existe la clase: {clase}")
        continue
    imagenes = obtener_imagenes(ruta_clase)
    random.shuffle(imagenes)
    total = len(imagenes)
    n_train = int(total * PORC_TRAIN)
    n_val = int(total * PORC_VAL)
    splits = {
    "train": imagenes[:n_train],
    "val": imagenes[n_train:n_train + n_val],
    "test": imagenes[n_train + n_val:]
    }
    print(f"\nClase: {clase}")
    print(f"Total origen: {total}")
    for split, archivos in splits.items():
        guardadas = 0
    for i, archivo in enumerate(archivos):
        destino = RUTA_FINAL / split / clase / f"{clase}_{i:05d}.jpg"
    if guardar_recorte(archivo, destino):
        guardadas += 1
    print(f"{split}: {guardadas}")
    print("\nDataset general preparado correctamente.")
if __name__ == "__main__":
    main()