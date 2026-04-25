# pylint: disable=line-too-long, missing-docstring
import os
import glob

import sys
import cv2

from ultralytics import YOLO

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_grayscale.py <path_to_video_or_image>")
        sys.exit(1)

    media_path = sys.argv[1]


    # Check for exported engine first, fallback to .pt
    train_dirs = glob.glob("runs/detect/train*")
    if not train_dirs:
        print("No trained model found. Please run train_yolo.py first.")
        sys.exit(1)

    latest_train_dir = max(train_dirs, key=os.path.getmtime)

    model_path = os.path.join(latest_train_dir, "weights", "best.engine")
    if not os.path.exists(model_path):
        model_path = os.path.join(latest_train_dir, "weights", "best.pt")
        if not os.path.exists(model_path):
            print("No weights found in latest run.")
            sys.exit(1)

    print(f"Loading model from {model_path}")
    model = YOLO(model_path, task='detect')

    # Ensure it's treated as an NMS-free check by examining the model architecture or output if needed
    # But YOLO class encapsulates this. The prompt states:
    # "ensure the input frame is converted to grayscale via OpenCV... before being passed"

    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        # Fallback to image reading
        frame = cv2.imread(media_path)
        if frame is None:
            print("Failed to load media.")
            sys.exit(1)

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run inference
        results = model(gray_frame)

        print(f"Detected {len(results[0].boxes)} objects.")
        print("Inference completed successfully without manual NMS (handled natively or NMS-free head).")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Inference
        results = model(gray_frame)

        # Display
        annotated_frame = results[0].plot()
        cv2.imshow("Grayscale Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
