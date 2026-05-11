#!/usr/bin/env python3
import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path


TARGET_FILENAME = "patchscopes_all_attention_mass.parquet"


def copy_file(src, dst):
    print(f"COPY {src} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_archive(staging_out_dir, output_archive):
    output_archive = output_archive.resolve()
    output_archive.parent.mkdir(parents=True, exist_ok=True)

    print(f"ARCHIVE {staging_out_dir} -> {output_archive}")

    with tarfile.open(output_archive, "w:gz") as tf:
        tf.add(staging_out_dir, arcname="out")

    return output_archive


def package_release(input_dir, output_archive):
    input_dir = input_dir.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        staging_out_dir = Path(tmp) / "out"

        for src in input_dir.rglob(TARGET_FILENAME):
            rel = src.relative_to(input_dir)
            dst = staging_out_dir / rel
            copy_file(src, dst)

        archive_path = make_archive(staging_out_dir, output_archive)
        print(f"Done: {archive_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default="./out", type=Path)
    parser.add_argument("--output_archive", required=True, type=Path)

    args = parser.parse_args()

    package_release(
        input_dir=args.input_dir,
        output_archive=args.output_archive,
    )


if __name__ == "__main__":
    main()
