# pylint: disable=line-too-long, missing-docstring
import os
import test_grayscale

if __name__ == "__main__":
    BASE_DIR = "drone-racing-dataset/autonomous"

    flights = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for flight in flights:
        media_file = BASE_DIR + "/" + flight + "/camera_" + flight
        output_path = os.path.join(OUTPUT_DIR, media_file.split("/")[-1] + "_output.mp4")
        test_grayscale.run(media_file, output_path)
