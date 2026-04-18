"""
process_generated_dataset.py
============================

Two operating modes:

  visualize   – Mirror the cluster folder structure but render each caption
                in a black strip appended below the image.

  build       – Create a self-contained VizWiz-format dataset directory from:
                  • an optional source VizWiz root (images copied via symlink)
                  • one or more generated-image directories

Usage examples
--------------
# Visualize both generated sets
python process_generated_dataset.py visualize \\
    --csv_path ../embeddings/cleaned_captions.csv \\
    --img_dir  ../visualizations/generated_clusters2S0CFG \\
    --out_dir  ../visualizations/generated_clusters2S0CFG_visualized

# Build VizWiz + 2S0CFG
python process_generated_dataset.py build \\
    --csv_path    ../embeddings/cleaned_captions.csv \\
    --out_root    /ghome/group01/C5/dataset/VizWiz_plus_2S0CFG \\
    --vizwiz_root /ghome/group01/C5/dataset/VizWiz \\
    --img_dirs    ../visualizations/generated_clusters2S0CFG

# Build generated-only (both sets)
python process_generated_dataset.py build \\
    --csv_path ../embeddings/cleaned_captions.csv \\
    --out_root /ghome/group01/C5/dataset/generated_both_only \\
    --img_dirs ../visualizations/generated_clusters2S0CFG \\
               ../visualizations/generated_clusters4S1CFG
"""

import os
import json
import argparse
import textwrap
import copy

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "cluster_id" in df.columns and "cluster" not in df.columns:
        df = df.rename(columns={"cluster_id": "cluster"})
    if "cluster" not in df.columns or "caption" not in df.columns:
        raise ValueError(
            f"CSV must have 'cluster' (or 'cluster_id') and 'caption' columns. "
            f"Found: {list(df.columns)}"
        )
    return df


def _build_caption_map(df: pd.DataFrame) -> dict:
    """Return {(cluster_id, local_idx): caption} preserving per-cluster order."""
    caption_map: dict = {}
    cluster_counters: dict = {}
    for _, row in df.iterrows():
        cid = int(row["cluster"])
        local_idx = cluster_counters.get(cid, 0)
        cluster_counters[cid] = local_idx + 1
        caption_map[(cid, local_idx)] = row["caption"]
    return caption_map


def _collect_images(img_dir: str) -> list:
    """Walk img_dir, return [(cluster_id, local_idx, path), ...]."""
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    entries = []
    for cluster_folder in sorted(os.listdir(img_dir)):
        cluster_path = os.path.join(img_dir, cluster_folder)
        if not os.path.isdir(cluster_path) or not cluster_folder.startswith("cluster_"):
            continue
        try:
            cluster_id = int(cluster_folder.split("_", 1)[1])
        except ValueError:
            continue
        local_idx = 0
        for fname in sorted(os.listdir(cluster_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                entries.append((cluster_id, local_idx, os.path.join(cluster_path, fname)))
                local_idx += 1
    return entries


def _get_font(font_size: int):
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_caption_on_image(img: Image.Image, caption: str) -> Image.Image:
    """Return image with caption in a black strip appended below (never clips)."""
    img = img.convert("RGB")
    W, H = img.size

    font_size = max(14, W // 40)
    font = _get_font(font_size)

    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    try:
        avg_char_w = dummy_draw.textlength("x", font=font)
    except AttributeError:
        avg_char_w = font_size * 0.6
    wrap_chars = max(20, int((W - 16) / max(avg_char_w, 1)))

    lines = textwrap.fill(caption, width=wrap_chars).splitlines()
    line_h = font_size + 6
    padding = 8
    strip_h = line_h * len(lines) + padding * 2

    out = Image.new("RGB", (W, H + strip_h), color=(0, 0, 0))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    y = H + padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    return out


# ---------------------------------------------------------------------------
# Mode: visualize
# ---------------------------------------------------------------------------

def mode_visualize(csv_path: str, img_dir: str, out_dir: str):
    """Render each caption below its image; save to mirrored folder structure."""
    df = _load_csv(csv_path)
    caption_map = _build_caption_map(df)
    entries = _collect_images(img_dir)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Loaded {len(entries)} images  from '{img_dir}'")
    print(f"Loaded {len(caption_map)} captions from '{csv_path}'")

    skipped = 0
    for cluster_id, local_idx, src_path in entries:
        caption = caption_map.get((cluster_id, local_idx))
        if caption is None:
            print(f"  [WARN] No caption for cluster={cluster_id} idx={local_idx} – skipping")
            skipped += 1
            continue

        try:
            img_out = _draw_caption_on_image(Image.open(src_path), caption)
        except Exception as e:
            print(f"  [WARN] Skipping corrupt image cluster={cluster_id} idx={local_idx}: {e}")
            skipped += 1
            continue
        cluster_out = os.path.join(out_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_out, exist_ok=True)
        fname = os.path.splitext(os.path.basename(src_path))[0] + ".jpg"
        img_out.save(os.path.join(cluster_out, fname), format="JPEG", quality=90)
        print(f"  cluster={cluster_id} idx={local_idx} saved")

    print(f"\nDone. {len(entries) - skipped} images saved to '{out_dir}'. "
          f"{skipped} skipped.")


# ---------------------------------------------------------------------------
# Mode: build
# ---------------------------------------------------------------------------

_EMPTY_ANNOTATION_BASE = {
    "info": {
        "description": "Synthetic VizWiz-format dataset for fine-tuning.",
        "version": "1.0",
    },
    "images": [],
    "annotations": [],
}


def _symlink_or_copy(src: str, dst: str):
    """Create a relative symlink from dst → src (fast). Falls back to copy."""
    if os.path.exists(dst) or os.path.islink(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def mode_build(
    csv_path: str,
    out_root: str,
    vizwiz_root: str | None,
    img_dirs: list[str],
):
    """
    Create a self-contained VizWiz-format dataset at out_root:
      out_root/images/train/  ← original (symlinked) + generated (copied as JPEG)
      out_root/annotations/train.json

    vizwiz_root : if given, the original VizWiz train split is included as base.
    img_dirs    : list of generated cluster dirs to append.
    """
    out_images_dir = os.path.join(out_root, "images", "train")
    out_ann_dir    = os.path.join(out_root, "annotations")
    out_ann_path   = os.path.join(out_ann_dir, "train.json")

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_ann_dir,    exist_ok=True)

    # ── Start from VizWiz base or empty skeleton ─────────────────────────────
    if vizwiz_root:
        src_ann_path = os.path.join(vizwiz_root, "annotations", "train.json")
        src_img_dir  = os.path.join(vizwiz_root, "images", "train")

        print(f"Loading VizWiz base from '{vizwiz_root}' …")
        with open(src_ann_path) as f:
            data = json.load(f)

        # Deep-copy so we don't mutate the loaded object
        data = copy.deepcopy(data)

        # Symlink every existing training image into the output folder
        orig_count = 0
        for img_rec in data["images"]:
            src = os.path.join(src_img_dir, img_rec["file_name"])
            dst = os.path.join(out_images_dir, img_rec["file_name"])
            _symlink_or_copy(src, dst)
            orig_count += 1

        print(f"  Symlinked {orig_count} original images.")
    else:
        data = copy.deepcopy(_EMPTY_ANNOTATION_BASE)
        print("No VizWiz base – starting from empty dataset.")

    existing_images      = data.get("images", [])
    existing_annotations = data.get("annotations", [])

    next_image_id = max((r["id"] for r in existing_images),      default=-1) + 1
    next_ann_id   = max((r["id"] for r in existing_annotations), default=-1) + 1

    # ── Append generated image dirs ────────────────────────────────────────
    if not img_dirs:
        print("No generated image dirs specified – writing annotation file only.")
    else:
        df = _load_csv(csv_path)
        caption_map = _build_caption_map(df)

        new_images      = []
        new_annotations = []

        for img_dir in img_dirs:
            entries = _collect_images(img_dir)
            print(f"\nProcessing '{img_dir}' ({len(entries)} images) …")
            skipped = 0

            for cluster_id, local_idx, src_path in entries:
                caption = caption_map.get((cluster_id, local_idx))
                if caption is None:
                    print(f"  [WARN] No caption cluster={cluster_id} idx={local_idx} – skip")
                    skipped += 1
                    continue

                new_fname = f"VizWiz_train_{next_image_id:08d}.jpg"
                dst_path  = os.path.join(out_images_dir, new_fname)
                try:
                    Image.open(src_path).convert("RGB").save(dst_path, format="JPEG", quality=92)
                except Exception as e:
                    print(f"  [WARN] Skipping corrupt image cluster={cluster_id} idx={local_idx}: {e}")
                    skipped += 1
                    continue

                new_images.append({
                    "file_name":     new_fname,
                    "id":            next_image_id,
                    "text_detected": False,
                    "vizwiz_url":    "",
                })
                new_annotations.append({
                    "id":       next_ann_id,
                    "image_id": next_image_id,
                    "caption":  caption,
                })

                next_image_id += 1
                next_ann_id   += 1

            added = len(entries) - skipped
            print(f"  Added {added} images, {added} annotations. "
                  f"({skipped} skipped)")

        data["images"]      = existing_images + new_images
        data["annotations"] = existing_annotations + new_annotations

    # ── Write annotation JSON ────────────────────────────────────────────────
    with open(out_ann_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"\n✓ Dataset written to '{out_root}'")
    print(f"  Total images:      {len(data['images'])}")
    print(f"  Total annotations: {len(data['annotations'])}")
    print(f"  Annotation file:   {out_ann_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- visualize ------------------------------------------------------------
    vis = sub.add_parser("visualize",
                         help="Render captions below images in a mirrored folder.")
    vis.add_argument("--csv_path", default="../embeddings/cleaned_captions.csv")
    vis.add_argument("--img_dir",  default="../visualizations/generated_clusters2S0CFG")
    vis.add_argument("--out_dir",  default="../visualizations/generated_visualized")

    # -- build ----------------------------------------------------------------
    bld = sub.add_parser("build",
                         help="Create a VizWiz-format dataset (base + generated).")
    bld.add_argument("--csv_path",    default="../embeddings/cleaned_captions.csv",
                     help="Path to cleaned_captions.csv")
    bld.add_argument("--out_root",    required=True,
                     help="Output dataset root directory")
    bld.add_argument("--vizwiz_root", default=None,
                     help="Source VizWiz root to use as base (omit for generated-only)")
    bld.add_argument("--img_dirs",    nargs="*", default=[],
                     help="One or more generated-cluster dirs to append")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "visualize":
        mode_visualize(csv_path=args.csv_path,
                       img_dir=args.img_dir,
                       out_dir=args.out_dir)
    elif args.mode == "build":
        mode_build(csv_path=args.csv_path,
                   out_root=args.out_root,
                   vizwiz_root=args.vizwiz_root,
                   img_dirs=args.img_dirs or [])


if __name__ == "__main__":
    main()
