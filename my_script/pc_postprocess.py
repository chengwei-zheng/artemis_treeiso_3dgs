import argparse
import colorsys
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


LABEL_GROUND = -1
LABEL_NOISE  = -2

COLOR_GROUND = np.array([160, 160, 160], dtype=np.uint8)  # medium gray
COLOR_NOISE  = np.array([50,  50,  50],  dtype=np.uint8)  # near-black


def load_xyz(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in ('.las', '.laz'):
        import laspy
        las = laspy.read(str(path))
        return np.vstack([las.x, las.y, las.z]).T
    elif suffix == '.ply':
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points)
    else:
        sys.exit(f"Unsupported format for xyz: {suffix}")


def write_ply_colored(xyz, colors, path):
    n = len(xyz)
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    data = np.empty(n, dtype=dtype)
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['red'], data['green'], data['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


def visualize_backproj(labels, xyz, out_path, seed=42):
    unique_tree_labels = np.unique(labels[labels >= 0])
    n = len(unique_tree_labels)
    rng = np.random.default_rng(seed)
    hues = rng.permutation(np.linspace(0, 1, n, endpoint=False))
    tree_colors = (np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues]) * 255).astype(np.uint8)
    label_to_color = dict(zip(unique_tree_labels, tree_colors))

    colors = np.empty((len(labels), 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        colors[labels == label] = color
    colors[labels == LABEL_GROUND] = COLOR_GROUND
    colors[labels == LABEL_NOISE]  = COLOR_NOISE

    write_ply_colored(xyz, colors, out_path)
    print(f"Saved: {out_path}")
1

def backproject(treeiso_laz_path, index_map_path, original_laz_path=None, knn_threshold=0.2):
    """Back-project treeiso final_segs labels onto the original (pre-preprocessing) point cloud.

    Label assignment rules:
      ground      — always LABEL_GROUND (-1)
      survivor    — treeiso final_segs label (>= 0)
      noise       — KNN from (survivors + ground); assign matched label if dist < knn_threshold, else LABEL_NOISE
      opacity_low — same rule as noise
      downsampled — KNN from (survivors + ground); always assign matched label (no threshold)
    """
    from scipy.spatial import cKDTree
    import laspy

    idx_map = np.load(index_map_path)
    survivor    = idx_map['survivor']
    ground      = idx_map['ground']
    noise       = idx_map['noise']
    opacity_low = idx_map['opacity_low'] if 'opacity_low' in idx_map else np.array([], dtype=np.int32)
    downsampled = idx_map['downsampled']

    n_original = len(survivor) + len(ground) + len(noise) + len(opacity_low) + len(downsampled)
    print(f"Original points: {n_original}  "
          f"(survivor={len(survivor)}, ground={len(ground)}, "
          f"noise={len(noise)}, opacity_low={len(opacity_low)}, downsampled={len(downsampled)})")

    las = laspy.read(str(treeiso_laz_path))
    final_segs = np.asarray(las.final_segs)   # length == len(survivor)

    # Initialize defaults
    labels = np.zeros(n_original, dtype=np.int32)
    labels[survivor]    = final_segs
    labels[ground]      = LABEL_GROUND   # always ground, never overridden
    labels[noise]       = LABEL_NOISE    # default; overridden by KNN if close enough
    labels[opacity_low] = LABEL_NOISE    # default; overridden by KNN if close enough

    needs_knn = len(noise) > 0 or len(opacity_low) > 0 or len(downsampled) > 0
    if needs_knn:
        if original_laz_path is None:
            sys.exit("Error: --original is required when noise/opacity/downsampled points exist")

        xyz_orig = load_xyz(original_laz_path)
        xyz_survivor = np.vstack([las.x, las.y, las.z]).T

        # KNN source: survivors + ground (all definitively labeled)
        if len(ground) > 0:
            xyz_source = np.vstack([xyz_survivor, xyz_orig[ground]])
            labels_source = np.concatenate([final_segs,
                                            np.full(len(ground), LABEL_GROUND, dtype=np.int32)])
        else:
            xyz_source = xyz_survivor
            labels_source = final_segs

        print(f"Building KD-tree from {len(xyz_source)} labeled points...")
        kdtree = cKDTree(xyz_source)

        # noise + opacity_low: assign only if within knn_threshold
        threshold_targets = np.concatenate([noise, opacity_low])
        if len(threshold_targets) > 0:
            print(f"KNN for {len(threshold_targets)} noise/opacity-filtered points "
                  f"(threshold={knn_threshold}m)...")
            dists, nn_idx = kdtree.query(xyz_orig[threshold_targets], k=1, workers=-1)
            mask = dists < knn_threshold
            labels[threshold_targets[mask]] = labels_source[nn_idx[mask]]
            print(f"  Reassigned: {mask.sum()},  kept as noise: {(~mask).sum()}")

        # downsampled: always assign nearest label
        if len(downsampled) > 0:
            print(f"KNN for {len(downsampled)} downsampled points...")
            _, nn_idx = kdtree.query(xyz_orig[downsampled], k=1, workers=-1)
            labels[downsampled] = labels_source[nn_idx]

    return labels


def main():
    parser = argparse.ArgumentParser(description="Post-processing for treeiso LAZ output")
    parser.add_argument("input", help="Treeiso output LAZ file (*_treeiso.laz)")
    parser.add_argument("--export-trees", type=int, default=None, metavar="N",
                        help="Export each final segment with >= N points as a separate PLY")

    # back-projection
    parser.add_argument("--backproject", action="store_true",
                        help="Back-project treeiso labels onto the original point cloud")
    parser.add_argument("--index-map", default=None,
                        help="Path to index_map.npz produced by pc_preprocess.py")
    parser.add_argument("--original", default=None,
                        help="Original input file (.las/.laz/.ply) before preprocessing")
    parser.add_argument("--knn-threshold", type=float, default=1.0,
                        help="Max distance (m) for noise/opacity-low points to adopt a neighbor's label (default: 1.0)")

    # visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Save a colored PLY of the back-projected labels")
    parser.add_argument("--backproj-npy", default=None,
                        help="Existing backproj.npy to visualize (used without --backproject)")

    args = parser.parse_args()

    las_path = Path(args.input)
    if not las_path.exists():
        sys.exit(f"Error: file not found: {las_path}")

    out_dir = las_path.parent

    if args.export_trees is not None:
        export_trees(las_path, args.export_trees, out_dir)

    labels = None

    if args.backproject:
        if args.index_map is None:
            sys.exit("Error: --index-map is required for --backproject")
        index_map_path = Path(args.index_map)
        if not index_map_path.exists():
            sys.exit(f"Error: file not found: {index_map_path}")
        original_path = Path(args.original) if args.original else None

        labels = backproject(las_path, index_map_path, original_path, args.knn_threshold)

        npy_path = out_dir / f"{las_path.stem}_backproj.npy"
        np.save(npy_path, labels)
        print(f"Saved: {npy_path}")

    if args.visualize:
        if args.original is None:
            sys.exit("Error: --original is required for --visualize")
        if labels is None:
            if args.backproj_npy is None:
                sys.exit("Error: --backproj-npy is required when --visualize is used without --backproject")
            labels = np.load(args.backproj_npy)
        xyz = load_xyz(args.original)
        vis_path = out_dir / f"{las_path.stem}_backproj_vis.ply"
        visualize_backproj(labels, xyz, vis_path)


if __name__ == "__main__":
    main()
