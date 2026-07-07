from ultralytics import YOLO
def main():
    modelo = YOLO("yolo26s-cls.pt")
    modelo.train(
    data="dataset_emociones_general/final",
    epochs=120,
    imgsz=160,
    batch=32,
    device=0,
    workers=4,
    cache=False,
    patience=25,
    cos_lr=True,
    optimizer="AdamW",
    lr0=0.0008,
    project="resultados",
    name="emociones_general_7clases_s_160",
    pretrained=True,
    save=True,
    plots=True
    )
if __name__ == "__main__":
    main()
