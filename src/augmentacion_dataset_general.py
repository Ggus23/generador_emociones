import cv2
import random
from pathlib import Path
import numpy as np
random.seed(42)
RUTA_TRAIN = Path("dataset_emociones_general/final/train")
CLASES = [
    "angry",
    "neutral",
    "sad",
    "surprise",
    "bored"
]
AUMENTOS_POR_IMAGEN = 2
def brillo_contraste(img):
    alpha = random.uniform(0.75, 1.25)
    beta = random.randint(-25, 25)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
def rotacion_leve(img):
    h, w = img.shape[:2]
    angulo = random.uniform(-8, 8)
    matriz = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)
    return cv2.warpAffine(img, matriz, (w, h), borderMode=cv2.BORDER_REFLECT)
def ruido_leve(img):
    ruido = np.random.normal(0, 5, img.shape).astype(np.int16)
    salida = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    return salida
def zoom_leve(img):
    h, w = img.shape[:2]
    escala = random.uniform(0.92, 1.08)
    nuevo_w = int(w * escala)
    nuevo_h = int(h * escala)
    img_zoom = cv2.resize(img, (nuevo_w, nuevo_h))
    if escala >= 1:
        x = (nuevo_w - w) // 2
        y = (nuevo_h - h) // 2
        return img_zoom[y:y+h, x:x+w]
    salida = np.zeros_like(img)
    x = (w - nuevo_w) // 2
    y = (h - nuevo_h) // 2
    salida[y:y+nuevo_h, x:x+nuevo_w] = img_zoom
    return salida
def aplicar_augmentacion(img):
    operaciones = [
    brillo_contraste,
    rotacion_leve,
    ruido_leve,
    zoom_leve
    ]
    random.shuffle(operaciones)
    salida = img.copy()
    for op in operaciones[:random.randint(2, 4)]:
        salida = op(salida)
    return salida
def main():
    for clase in CLASES:
        ruta_clase = RUTA_TRAIN / clase
        if not ruta_clase.exists():
            print(f"No existe: {ruta_clase}")
        continue
    imagenes = [
    p for p in ruta_clase.iterdir()
    if p.is_file()
    and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    and "_aug_" not in p.stem
    ]
    print(f"\nClase: {clase}")
    print(f"Originales: {len(imagenes)}")
    creadas = 0
    for imagen_path in imagenes:
        img = cv2.imread(str(imagen_path))
        if img is None:
            continue
    for i in range(AUMENTOS_POR_IMAGEN):
        aumentada = aplicar_augmentacion(img)
    destino = ruta_clase / f"{imagen_path.stem}_aug_{i}.jpg"
    cv2.imwrite(str(destino), aumentada)
    creadas += 1
    print(f"Aumentadas: {creadas}")
    print("\nAugmentacion terminada.")
if __name__ == "__main__":
    main()