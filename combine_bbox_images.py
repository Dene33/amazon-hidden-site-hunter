import argparse
from pathlib import Path
from sentinel_utils import mosaic_images


def combine_from_dir(out_dir: Path) -> None:
    """Create mosaics for images with the same name across bbox folders."""
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)

    bbox_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    images_by_name = {}
    for bdir in bbox_dirs:
        for p in bdir.glob('*.[jp][pn]g'):
            images_by_name.setdefault(p.name, []).append(p)

    combined_dir = out_dir / 'combined'
    combined_dir.mkdir(exist_ok=True)
    for name, paths in images_by_name.items():
        if len(paths) < 2:
            continue
        out_path = combined_dir / name
        mosaic_images(paths, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine images across bbox outputs")
    ap.add_argument('out_dir', help='Path to directory containing bbox folders')
    args = ap.parse_args()
    combine_from_dir(Path(args.out_dir))


if __name__ == '__main__':
    main()
