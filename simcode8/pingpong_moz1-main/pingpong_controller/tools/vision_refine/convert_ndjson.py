import asyncio
import argparse
from pathlib import Path

from ultralytics.data.converter import convert_ndjson_to_yolo


def main():
    parser = argparse.ArgumentParser(
        description="Convert Label Studio NDJSON annotations to YOLO format."
    )
    parser.add_argument(
        "ndjson_path",
        help="Path to the input .ndjson file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dataset directory. Defaults to a folder next to the NDJSON file.",
    )
    args = parser.parse_args()

    ndjson_path = Path(args.ndjson_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else ndjson_path.with_suffix("")
    )

    yaml_path = asyncio.run(
        convert_ndjson_to_yolo(str(ndjson_path), output_path=str(output_dir))
    )

    print("Generated data.yaml:", yaml_path)


if __name__ == "__main__":
    main()
