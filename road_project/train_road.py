from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train(model):

    model.train(
        data="./dataset/dataset.yaml",
        epochs=20, batch=32, workers=0,
        device='cuda'
    )


if __name__ == '__main__':
    # model = YOLO("./models/yolo11n.pt")
    model = YOLO("./yolo11n.pt")
    train(model)
