from ultralytics import YOLO
def main():
    modelo = YOLO("runs/classify/resultados/emociones_general_7clases_s_160/weights/best.pt")
    metricas = modelo.val(data="dataset_emociones_general/final")
    print(metricas)
if __name__ == "__main__":
    main()
