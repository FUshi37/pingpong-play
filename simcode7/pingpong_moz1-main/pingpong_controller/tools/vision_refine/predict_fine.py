import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Run visual prediction checks for a fine-tuned YOLO segmentation model."
    )
    parser.add_argument(
        "model_path",
        help="Path to the YOLO model weights, for example weights/best.pt.",
    )
    parser.add_argument(
        "source",
        help="Image file or image directory to run prediction on.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Prediction output directory. Defaults to outputs/vision_refine/predict.",
    )
    parser.add_argument(
        "--name",
        default="predict_finetune",
        help="Prediction run name.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=848,
        help="Prediction image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Prediction device, for example 0 or cpu. Defaults to Ultralytics auto selection.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw labels on saved prediction images.",
    )
    parser.add_argument(
        "--show-conf",
        action="store_true",
        help="Draw confidence values on saved prediction images.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reusing an existing output run directory.",
    )
    args = parser.parse_args()

    current_file = Path(__file__).resolve()
    package_dir = current_file.parent.parent.parent
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else package_dir / "outputs" / "vision_refine" / "predict"
    )

    predict_kwargs = {
        "source": str(Path(args.source).expanduser().resolve()),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "project": str(output_dir),
        "name": args.name,
        "save": True,
        "show_labels": args.show_labels,
        "show_conf": args.show_conf,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    model = YOLO(str(Path(args.model_path).expanduser().resolve()))
    results = model.predict(**predict_kwargs)
    save_dir = getattr(results[0], "save_dir", None) if results else None
    print("Prediction complete:", save_dir or output_dir / args.name)


if __name__ == "__main__":
    main()
