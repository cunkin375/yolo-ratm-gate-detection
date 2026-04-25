# pylint: disable=line-too-long, missing-docstring
import os
import cv2
import numpy as np

def create_mock_flight(base_path, flight_name, num_frames=5):
    img_dir = os.path.join(base_path, flight_name, f"camera_{flight_name}")
    lbl_dir = os.path.join(base_path, flight_name, f"label_{flight_name}")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(num_frames):
        # Create a mock grayscale image (saved as BGR for consistency)
        img = np.ones((416, 416, 3), dtype=np.uint8) * (50 + i * 10)
        img_name = f"frame_{i:04d}.jpg"
        cv2.imwrite(os.path.join(img_dir, img_name), img)

        # Create mock label: class cx cy w h kps...
        lbl_name = f"frame_{i:04d}.txt"
        with open(os.path.join(lbl_dir, lbl_name), "w") as f:
            # Fake bounding box and keypoints
            f.write("0 0.5 0.5 0.2 0.2 0 0 0 0 0 0 0 0 0 0 0 0\n")

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "drone-racing-dataset", "data")

    # Autonomous flights
    auto_dir = os.path.join(data_dir, "autonomous")
    for i in range(4):
        create_mock_flight(auto_dir, f"flight-{i:02d}a-mock")

    # Piloted flights (used as val roughly since 1 out of 5 is val in my logic)
    piloted_dir = os.path.join(data_dir, "piloted")
    for i in range(2):
        create_mock_flight(piloted_dir, f"flight-{i:02d}p-mock")

    print("Mock data generated.")

if __name__ == "__main__":
    main()
