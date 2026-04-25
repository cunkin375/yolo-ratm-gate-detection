# pylint: disable=line-too-long, missing-docstring
import sys
from ultralytics import YOLO

def main():
    # Load the Nano variant of YOLO26 (NMS-free)
    try:
        model = YOLO("yolo26n.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Execute training
    model.train(
        data="drl_gates.yaml",
        epochs=50,
        save_period=10,        # Save model every 10 epochs
        imgsz=640,
        batch=-1,             # Use Ultralytics auto batch feature
        optimizer="MuSGD",    # 2026 Hybrid optimizer
        augment=True,
        cache=True,           # Speed up training on Windows SSDs
        device=0,             # Target first NVIDIA GPU
        val=True
    )

    # Export to pytorch for easy deployment
    model.export(format="torchscript", imgsz=640)

    # Export to TensorRT for deployment
    # model.export(format="engine", half=True, imgsz=640)

if __name__ == "__main__":
    main()
