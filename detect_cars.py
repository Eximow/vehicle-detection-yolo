import cv2
from ultralytics import YOLO
import sys
from pathlib import Path

def detect_cars(input_image_path: str, output_image_path: str = "output.jpg", conf_threshold: float = 0.25):
    # Load pre-trained YOLO model
    model = YOLO("yolov8n.pt")  # You can also use larger versions like yolov8s.pt (more accurate but slower)

    # Read input image
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_image_path}")

    # Run detection
    results = model(img, conf=conf_threshold)[0]

    # COCO classes for vehicles
    vehicle_classes = {2, 3, 5, 7}  # 2: car, 3: motorcycle, 5: bus, 7: truck

    count = 0
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        cls_id = int(cls.item())
        if cls_id in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.tolist())
            # Draw bounding box and label
            label = f"{results.names[cls_id]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            count += 1

    # Write total count on the image
    cv2.putText(img, f"Vehicles: {count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 255), 2)

    # Save output image
    cv2.imwrite(output_image_path, img)
    print(f"Number of vehicles: {count}")
    print(f"Output image saved: {output_image_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_cars.py <input_image_path> [output_image_path] [confidence_threshold]")
        print("Example: python detect_cars.py street.jpg output.jpg 0.3")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "output.jpg"
