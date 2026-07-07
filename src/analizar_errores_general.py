from pathlib import Path
import shutil
from collections import Counter
from ultralytics import YOLO
RUTA_MODELO = "runs/classify/resultados/emociones_general_7clases_s_160/weights/best.pt"
RUTA_TEST = Path("dataset_emociones_general/final/test")
RUTA_SALIDA = Path("analisis_errores_general")
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def obtener_imagenes(carpeta):
    return [p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES]
def main():
    modelo = YOLO(RUTA_MODELO)
    if RUTA_SALIDA.exists():
        shutil.rmtree(RUTA_SALIDA)
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    total = 0
    correctas = 0
    errores_por_clase = Counter()
    confusiones = Counter()
    for carpeta_clase in sorted([p for p in RUTA_TEST.iterdir() if p.is_dir()]):
        clase_real = carpeta_clase.name
        imagenes = obtener_imagenes(carpeta_clase)
    for imagen in imagenes:
        resultados = modelo.predict(source=str(imagen), imgsz=160, verbose=False)
        if not resultados:
            continue
    r = resultados[0]
    probs = r.probs
    clase_predicha = r.names[probs.top1]
    confianza = float(probs.top1conf)
    total += 1
    if clase_predicha == clase_real:
        correctas += 1
    else:
        errores_por_clase[clase_real] += 1
    confusiones[(clase_real, clase_predicha)] += 1
    carpeta_error = RUTA_SALIDA / f"{clase_real}_como_{clase_predicha}"
    carpeta_error.mkdir(parents=True, exist_ok=True)
    nombre_nuevo = f"{imagen.stem}_conf_{confianza:.2f}{imagen.suffix}"
    shutil.copy2(imagen, carpeta_error / nombre_nuevo)
    accuracy = correctas / total if total > 0 else 0
    print(f"Total evaluadas: {total}")
    print(f"Correctas: {correctas}")
    print(f"Accuracy manual: {accuracy:.4f}")
    print("\nErrores por clase real:")
    for clase, cantidad in errores_por_clase.most_common():
        print(f"{clase}: {cantidad}")
    print("\nConfusiones principales:")
    for (real, pred), cantidad in confusiones.most_common(20):
        print(f"{real} -> {pred}: {cantidad}")
    print(f"\nErrores guardados en: {RUTA_SALIDA}")
if __name__ == "__main__":
    main()
