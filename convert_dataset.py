# pylint: disable=line-too-long, missing-docstring
import os
import glob
import shutil
import cv2

def convert_labels(src_label_dir, dest_label_dir):
    os.makedirs(dest_label_dir, exist_ok=True)
    for txt_file in glob.glob(os.path.join(src_label_dir, "*.txt")):
        basename = os.path.basename(txt_file)
        dest_file = os.path.join(dest_label_dir, basename)
        with open(txt_file, "r", encoding="utf-8") as f_in, open(dest_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class x_center y_center width height
                    f_out.write(" ".join(parts[:5]) + "\n")

def convert_images_to_grayscale(src_image_dir, dest_image_dir):
    os.makedirs(dest_image_dir, exist_ok=True)
    for img_file in glob.glob(os.path.join(src_image_dir, "*.*")):
        basename = os.path.basename(img_file)
        dest_file = os.path.join(dest_image_dir, basename)
        if not os.path.exists(dest_file):
            img = cv2.imread(img_file)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(dest_file, gray)
            else:
                print(f"Warning: Failed to read image {img_file}")

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "drone-racing-dataset")

    dataset_dir = os.path.join(base_dir, "dataset")
    train_images = os.path.join(dataset_dir, "images", "train")
    val_images = os.path.join(dataset_dir, "images", "val")
    train_labels = os.path.join(dataset_dir, "labels", "train")
    val_labels = os.path.join(dataset_dir, "labels", "val")

    os.makedirs(train_images, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)

    # Process flights
    flights = []
    for mode in ["autonomous", "piloted"]:
        mode_dir = os.path.join(data_dir, mode)
        if os.path.isdir(mode_dir):
            for flight_dir in os.listdir(mode_dir):
                full_flight_dir = os.path.join(mode_dir, flight_dir)
                if os.path.isdir(full_flight_dir):
                    flights.append(full_flight_dir)

    print(f"Found {len(flights)} flights.")

    # Simple split: 80% train, 20% val
    for i, flight_dir in enumerate(flights):
        flight_name = os.path.basename(flight_dir)

        # Determine paths based on label_visualization.py
        # Fallback to labels_ if label_ is not found
        img_dir = os.path.join(flight_dir, "camera_" + flight_name)
        lbl_dir = os.path.join(flight_dir, "label_" + flight_name)
        if not os.path.isdir(lbl_dir):
            lbl_dir = os.path.join(flight_dir, "labels_" + flight_name)

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            print(f"Skipping {flight_name}, missing images or labels")
            continue

        is_val = i % 5 == 0  # every 5th flight is validation

        dest_img = val_images if is_val else train_images
        dest_lbl = val_labels if is_val else train_labels

        print(f"Processing {flight_name} -> {'val' if is_val else 'train'}")
        convert_images_to_grayscale(img_dir, dest_img)
        convert_labels(lbl_dir, dest_lbl)

if __name__ == "__main__":
    main()
