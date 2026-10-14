import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Validate a fine-tuned YOLO segmentation model."
    )
    parser.add_argument(
        "model_path",
        help="Path to the YOLO model weights, for example weights/best.pt.",
    )
    parser.add_argument(
        "data_yaml",
        help="Path to the dataset data.yaml file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Validation output directory. Defaults to outputs/vision_refine/val.",
    )
    parser.add_argument(
        "--name",
        default="validate_finetune",
        help="Validation run name.",
    )
    args = parser.parse_args()

    current_file = Path(__file__).resolve()
    package_dir = current_file.parent.parent.parent
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else package_dir / "outputs" / "vision_refine" / "val"
    )

    model = YOLO(str(Path(args.model_path).expanduser().resolve()))
    metrics = model.val(
        data=str(Path(args.data_yaml).expanduser().resolve()),
        project=str(output_dir),
        name=args.name,
    )

    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)

    print("mAP50-95 (segmentation):", metrics.seg.map)
    print("mAP50 (segmentation):", metrics.seg.map50)


if __name__ == "__main__":
    main()
