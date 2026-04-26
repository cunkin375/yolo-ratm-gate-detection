# pylint: disable=line-too-long, missing-docstring
import os
import glob

import sys
import cv2

from ultralytics import YOLO

def run(input_media: str, output_path="ouptut_inference.mp4"):
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

    if os.path.isdir(input_media):
        # Handle directory of images
        image_files = sorted([
            os.path.join(input_media, f) for f in os.listdir(input_media)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not image_files:
            print(f"No images found in directory: {input_media}")
            sys.exit(1)

        print(f"Found {len(image_files)} images in directory. Processing...")

        first_frame = cv2.imread(image_files[0])
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # The model expects 3 channels, so we convert the 1-channel grayscale image back to a 3-channel format
            # where all channels have the same grayscale value.
            gray_3c = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # Inference
            results = model(gray_3c)

            # Display
            annotated_frame = results[0].plot()
            out_video.write(annotated_frame)
            cv2.imshow("Grayscale Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out_video.release()
        print(f"Video saved to {output_path}")

    else:
        cap = cv2.VideoCapture(input_media)
        if not cap.isOpened():
            # Fallback to image reading
            frame = cv2.imread(input_media)
            if frame is None:
                print("Failed to load media.")
                sys.exit(1)

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3c = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # Run inference
            results = model(gray_3c)

            while True:
                results[0].show()   # Open display window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            results[0].save("grayscale_output.jpg")

            print(f"Detected {len(results[0].boxes)} objects.")
            print("Inference completed successfully without manual NMS (handled natively or NMS-free head).")
            return

        # Handle video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps: # Check for 0 or NaN
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3c = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # Inference
            results = model(gray_3c)

            # Display
            annotated_frame = results[0].plot()
            out_video.write(annotated_frame)
            cv2.imshow("Grayscale Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out_video.release()
        print(f"Video saved to {output_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_grayscale.py <path_to_video_or_image>")
        sys.exit(1)

    media_path = sys.argv[1]

    run(media_path)
