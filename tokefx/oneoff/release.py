#!/usr/bin/env python3
import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

import pandas as pd


def copy_file(src, dst):
    print(f"COPY {src} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def filter_heads_file(src, dst, model, language):
    print(f"FILTER {src} -> {dst}  model={model!r} language={language!r}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)
    filtered = df[(df["model"] == model) & (df["lang"] == language)]
    filtered.to_parquet(dst, index=False)


def make_archive(staging_out_dir, output_archive):
    output_archive = output_archive.resolve()
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"ARCHIVE {staging_out_dir} -> {output_archive}")
    with tarfile.open(output_archive, "w:gz") as tf:
        tf.add(staging_out_dir, arcname="out")
    return output_archive


def package_release(input_dir, output_archive, model, language):
    input_dir = input_dir.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        staging_out_dir = Path(tmp) / "out"
        for src in input_dir.rglob("*.parquet"):
            rel = src.relative_to(input_dir)
            dst = staging_out_dir / rel
            if src.name == "full_attn_heads_all.parquet":
                filter_heads_file(
                    src,
                    dst,
                    model=model,
                    language=language,
                )
            else:
                copy_file(src, dst)
        archive_path = make_archive(staging_out_dir, output_archive)
        print(f"Done: {archive_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./out", type=Path)
    parser.add_argument("--output_archive", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()

    package_release(
        input_dir=args.input_dir,
        output_archive=args.output_archive,
        model=args.model,
        language=args.language,
    )


if __name__ == "__main__":
    main()
