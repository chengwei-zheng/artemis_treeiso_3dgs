import argparse
import sys
from pathlib import Path
import numpy as np


def write_ply(pts, path):
    pts = pts.astype(np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(pts.tobytes())


def export_trees(las_path, min_points, out_dir):
    import laspy
    las = laspy.read(str(las_path))
    xyz = np.vstack([las.x, las.y, las.z]).T
    labels = np.asarray(las.final_segs)

    trees_dir = out_dir / "trees"
    trees_dir.mkdir(exist_ok=True)
    exported = 0
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() < min_points:
            continue
        write_ply(xyz[mask], trees_dir / f"tree_{label:04d}.ply")
        exported += 1
    print(f"Exported {exported} trees (>={min_points} pts) to {trees_dir}")


def main():
    parser = argparse.ArgumentParser(description="Post-processing for treeiso LAZ output")
    parser.add_argument("input", help="Treeiso output LAZ file (*_treeiso.laz)")
    parser.add_argument("--export-trees", type=int, default=None, metavar="N",
                        help="Export each final segment with >= N points as a separate PLY")
    args = parser.parse_args()

    las_path = Path(args.input)
    if not las_path.exists():
        sys.exit(f"Error: file not found: {las_path}")

    out_dir = las_path.parent

    if args.export_trees is not None:
        export_trees(las_path, args.export_trees, out_dir)


if __name__ == "__main__":
    main()
