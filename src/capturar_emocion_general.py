import cv2
import argparse
from pathlib import Path
from datetime import datetime

CLASES_VALIDAS = [
    "angry",
    "neutral",
    "sad",
    "surprise",
    "bored"
]

parser = argparse.ArgumentParser()
parser.add_argument("--clase", required=True)
parser.add_argument("--persona", required=True)
args = parser.parse_args()

clase = args.clase.lower()
persona = args.persona.lower()

if clase not in CLASES_VALIDAS:
    raise ValueError(f"Clase no valida: {clase}")

# CORREGIDO: Ahora usa tu ruta personalizada
ruta_salida = Path("optencion_dataset") / "capturas" / clase
ruta_salida.mkdir(parents=True, exist_ok=True)

camara = cv2.VideoCapture(0)
if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la camara.")

contador = len(list(ruta_salida.glob("*.jpg")))

print(f"Capturando emocion: {clase}")
print(f"Persona: {persona}")
print("Presiona S para guardar.")
print("Presiona Q para salir.")

while True:
    ok, frame = camara.read()
    if not ok:
        break
        
    frame_mostrar = frame.copy()
    cv2.putText(
        frame_mostrar,
        f"Clase: {clase} | Persona: {persona} | Img: {contador}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    cv2.imshow("Captura dataset emociones", frame_mostrar)
    
    tecla = cv2.waitKey(1) & 0xFF
    
    # CORREGIDO: Todo este bloque ahora depende correctamente de presionar la 'S'
    if tecla == ord("s"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nombre = f"{clase}_{persona}_{timestamp}.jpg"
        destino = ruta_salida / nombre
        cv2.imwrite(str(destino), frame)
        print(f"Guardada: {destino}")
        contador += 1
        
    # CORREGIDO: El alineamiento con el 'if' es el correcto dentro del bucle
    elif tecla == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()