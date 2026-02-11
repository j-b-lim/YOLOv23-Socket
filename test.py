from ultralytics import YOLO


def main():
    model = YOLO("best.pt")

    # train_results = model.train(
    #     data="coco8.yaml",
    #     epochs=50,
    #     imgsz=640,
    #     device=0,
    #     workers=4,   # 있어도 OK
    # )
    # #
    # metrics = model.val(device=0)

    results = model(r"C:\Users\splab\Desktop\TEST_008.jpg")
    results[0].show()

    model.export(format="onnx")


if __name__ == "__main__":
    main()
