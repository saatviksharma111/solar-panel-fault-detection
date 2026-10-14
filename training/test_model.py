from ultralytics import YOLO

model = YOLO("../models/best_yolo26.pt")
results = model.predict(source="test_images", conf=0.4, save=True)

for r in results:
    print(r.path)
    for box in r.boxes:
        print("  ", model.names[int(box.cls)], round(float(box.conf), 2))