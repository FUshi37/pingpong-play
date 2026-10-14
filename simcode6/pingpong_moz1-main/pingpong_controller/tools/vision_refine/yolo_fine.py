import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO segmentation model for ping-pong detection."
    )
    parser.add_argument(
        "data_yaml",
        help="Path to the YOLO dataset data.yaml file.",
    )
    parser.add_argument(
        "--base-model",
        default="yolo11s-seg.pt",
        help="Base YOLO segmentation model or checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=848,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example 0 or cpu. Defaults to Ultralytics auto selection.",
    )
    parser.add_argument(
        "--name",
        default="yoloreal_finetune",
        help="Training run name.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training output directory. Defaults to outputs/vision_refine/runs.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision training.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable dataset caching.",
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
        else package_dir / "outputs" / "vision_refine" / "runs"
    )

    train_kwargs = {
        "data": str(Path(args.data_yaml).expanduser().resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "cache": args.cache,
        "amp": not args.no_amp,
        "project": str(output_dir),
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model = YOLO(args.base_model)
    results = model.train(**train_kwargs)
    print("Training complete:", results.save_dir)


if __name__ == "__main__":
    main()
