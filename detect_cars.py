import cv2
from ultralytics import YOLO
import sys
from pathlib import Path

def detect_cars(input_image_path: str, output_image_path: str = "output.jpg", conf_threshold: float = 0.25):
    # بارگذاری مدل از پیش‌آموزش‌داده‌شده
    model = YOLO("yolov8n.pt")  # می‌تونی نسخه‌های بزرگ‌تر مثل yolov8s.pt هم استفاده کنی (دقیق‌تر، اما کندتر)

    # خواندن تصویر
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"تصویر پیدا نشد: {input_image_path}")

    # اجرای تشخیص
    results = model(img, conf=conf_threshold)[0]

    # کلاس‌های COCO برای وسایل نقلیه
    vehicle_classes = {2, 3, 5, 7}  # 2: car, 3: motorcycle, 5: bus, 7: truck

    count = 0
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        cls_id = int(cls.item())
        if cls_id in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.tolist())
            # رسم مستطیل و برچسب
            label = f"{results.names[cls_id]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            count += 1

    # نوشتن تعداد روی تصویر
    cv2.putText(img, f"Vehicles: {count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 255), 2)

    # ذخیره خروجی
    cv2.imwrite(output_image_path, img)
    print(f"تعداد وسایل نقلیه: {count}")
    print(f"تصویر خروجی ذخیره شد: {output_image_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("نحوه اجرا: python detect_cars.py <مسیر_تصویر> [مسیر_خروجی] [آستانه_اعتماد]")
        print("مثال: python detect_cars.py street.jpg output.jpg 0.3")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "output.jpg"
    conf = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.25

    detect_cars(input_path, output_path, conf)
