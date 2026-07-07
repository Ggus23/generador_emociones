import cv2
from collections import Counter, deque
from ultralytics import YOLO

# CORREGIDO: Ruta en una sola línea continua
RUTA_MODELO = "runs\\classify\\resultados\\emociones_general_7clases_s_160\\weights\\best.pt"
CAMARA_INDEX = 0
CONFIANZA_MINIMA = 0.85
DIFERENCIA_MINIMA_TOP2 = 0.20
VENTANA_ESTABILIDAD = 15
PORCENTAJE_MINIMO_ESTABILIDAD = 0.80
ANCHO_MINIMO_ROSTRO = 90
ALTO_MINIMO_ROSTRO = 90
RUTA_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

NOMBRES_EMOCIONES = {
    "angry": "Enojado",
    "neutral": "Neutral",
    "sad": "Triste",
    "surprise": "Sorpresa",
    "bored": "Desinteres"
}

COLORES_EMOCIONES = {
    "angry": (0, 0, 255),
    "neutral": (255, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (0, 165, 255),
    "bored": (180, 180, 180),
    "sin dato": (120, 120, 120),
    "sin rostro": (120, 120, 120)
}

def obtener_top_probabilidades(resultado):
    probabilidades = resultado.probs.data.tolist()
    nombres = resultado.names
    pares = [(nombres[i], float(probabilidades[i])) for i in range(len(probabilidades))]
    pares.sort(key=lambda x: x[1], reverse=True)
    return pares

def obtener_prediccion_estable(historial):
    if len(historial) < VENTANA_ESTABILIDAD:
        return "sin dato", 0.0, "historial_insuficiente"
    conteo = Counter([item["emocion"] for item in historial])
    emocion_mas_frecuente, repeticiones = conteo.most_common(1)[0]
    minimo = int(VENTANA_ESTABILIDAD * PORCENTAJE_MINIMO_ESTABILIDAD)
    if repeticiones < minimo:
        return "sin dato", 0.0, "emocion_inestable"
    muestras = [item for item in historial if item["emocion"] == emocion_mas_frecuente]
    confianza_promedio = sum(item["confianza"] for item in muestras) / len(muestras)
    if confianza_promedio < CONFIANZA_MINIMA:
        return "sin dato", confianza_promedio, "baja_confianza"
    diferencias = []
    for item in muestras:
        top2 = item["top2"]
        if len(top2) >= 2:
            diferencias.append(top2[0][1] - top2[1][1])
    diferencia_promedio = sum(diferencias) / len(diferencias) if diferencias else 0.0
    if diferencia_promedio < DIFERENCIA_MINIMA_TOP2:
        return "sin dato", confianza_promedio, "top1_top2_cercanos"
    return emocion_mas_frecuente, confianza_promedio, "decision_estable"

def dibujar_panel(frame, emocion, confianza, motivo, top3):
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (620, 220), (25, 25, 25), -1)
    frame[:] = cv2.addWeighted(overlay, 0.38, frame, 0.62, 0)
    nombre = NOMBRES_EMOCIONES.get(emocion, emocion)
    color = COLORES_EMOCIONES.get(emocion, (255, 255, 255))
    y = 35
    cv2.putText(frame, "Reconocimiento de emociones", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,255,255), 2)
    y += 35
    cv2.putText(frame, f"Emocion estable: {nombre}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    y += 30
    cv2.putText(frame, f"Confianza: {confianza:.2f}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
    y += 28
    cv2.putText(frame, f"Motivo: {motivo}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)
    
    # CORREGIDO: Indentación interna del panel Top3
    if top3:
        y += 28
        texto = "Top3: " + " | ".join([f"{NOMBRES_EMOCIONES.get(n, n)} {p:.2f}" for n, p in top3])
        cv2.putText(frame, texto, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,255,180), 2)

def main():
    modelo = YOLO(RUTA_MODELO)
    detector_rostro = cv2.CascadeClassifier(RUTA_CASCADE)
    camara = cv2.VideoCapture(CAMARA_INDEX)
    
    # CORREGIDO: Retorno indentado dentro del condicional de error
    if not camara.isOpened():
        print("No se pudo abrir la camara.")
        return

    # CORREGIDO: Todo el bucle principal ahora pertenece formalmente a main()
    historial = deque(maxlen=VENTANA_ESTABILIDAD)
    print("Camara iniciada. Presiona q para salir.")
    
    while True:
        ok, frame = camara.read()
        if not ok:
            break
            
        frame_anotado = frame.copy()
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = detector_rostro.detectMultiScale(gris, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
        
        emocion_estable = "sin rostro"
        confianza_estable = 0.0
        motivo = "sin_rostro"
        top3 = []
        
        # CORREGIDO: Procesamiento seguro protegido por la detección de rostros
        if len(rostros) > 0:
            rostros = sorted(rostros, key=lambda r: r[2] * r[3], reverse=True)
            x, y, w, h = rostros[0]
            rostro = frame[y:y+h, x:x+w]
            
            resultados = modelo.predict(source=rostro, imgsz=160, verbose=False)
            if resultados:
                pares = obtener_top_probabilidades(resultados[0])
                emocion = pares[0][0]
                confianza = pares[0][1]
                top2 = pares[:2]
                top3 = pares[:3]
                
                if confianza >= 0.50:
                    historial.append({"emocion": emocion, "confianza": confianza, "top2": top2})
            
            emocion_estable, confianza_estable, motivo = obtener_prediccion_estable(historial)
            color = COLORES_EMOCIONES.get(emocion_estable, (0, 255, 0))
            cv2.rectangle(frame_anotado, (x, y), (x+w, y+h), color, 2)
            
            # CORREGIDO: String en una sola línea para evitar errores de sintaxis
            texto_pantalla = f"{NOMBRES_EMOCIONES.get(emocion_estable, emocion_estable)} {confianza_estable:.2f}"
            cv2.putText(frame_anotado, texto_pantalla, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        
        dibujar_panel(frame_anotado, emocion_estable, confianza_estable, motivo, top3)
        cv2.imshow("Detector de emociones", frame_anotado)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    camara.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()