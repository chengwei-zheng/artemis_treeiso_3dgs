import argparse
import json
import sys
from pathlib import Path
import numpy as np
import open3d as o3d


def load_las(path):
    import laspy
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd, las


def save_las(las, indices, out_path):
    import laspy
    mask = np.zeros(len(las.points), dtype=bool)
    mask[indices] = True
    out = laspy.LasData(header=las.header)
    out.points = las.points[mask]
    out.write(out_path)


def estimate_spacing(pcd, sample_size=5000):
    pts = np.asarray(pcd.points)
    if len(pts) > sample_size:
        idx = np.random.choice(len(pts), sample_size, replace=False)
        sampled = pcd.select_by_index(idx)
    else:
        sampled = pcd
    dists = np.asarray(sampled.compute_nearest_neighbor_distance())
    print(f"Point spacing estimate (avg nearest-neighbor, n={len(dists)}):")
    print(f"  mean={dists.mean():.4f}m  median={np.median(dists):.4f}m  "
          f"min={dists.min():.4f}m  max={dists.max():.4f}m")


def save_ply_as_laz(pcd, out_path):
    import laspy
    xyz = np.asarray(pcd.points)
    header = laspy.LasHeader(point_format=0, version="1.4")
    header.offsets = np.min(xyz, axis=0)
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    out = laspy.LasData(header=header)
    out.x = xyz[:, 0]
    out.y = xyz[:, 1]
    out.z = xyz[:, 2]
    out.write(str(out_path))


def remove_ground_csf(pcd, cloth_resolution, class_threshold, rigidness, export_cloth=False):
    import CSF
    csf = CSF.CSF()
    csf.params.cloth_resolution = cloth_resolution
    csf.params.class_threshold = class_threshold
    csf.params.rigidness = rigidness
    csf.params.bSloopSmooth = False
    csf.params.interations = 500

    csf.setPointCloud(np.asarray(pcd.points))
    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx)

    cloth_data = csf.do_cloth_export() if export_cloth else None

    non_ground_idx = list(non_ground_idx)
    print(f"Ground removal (CSF): {len(ground_idx)} ground / "
          f"{len(non_ground_idx)} non-ground out of {len(pcd.points)} total")
    return non_ground_idx, cloth_data


def save_ground_obj(cloth_tuple, out_path):
    pts = np.array(cloth_tuple).reshape(-1, 3)
    width = int(np.sum(pts[:, 1] == pts[0, 1]))
    height = len(pts) // width

    R, C = np.meshgrid(np.arange(height - 1), np.arange(width - 1), indexing='ij')
    R, C = R.ravel(), C.ravel()
    # quad faces (1-indexed): v00, v10, v11, v01 — counter-clockwise
    quads = np.stack([
        R * width + C,
        R * width + C + 1,
        (R + 1) * width + C + 1,
        (R + 1) * width + C,
    ], axis=1) + 1

    with open(out_path, 'w') as f:
        for p in pts:
            f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        for q in quads:
            f.write(f"f {q[0]} {q[1]} {q[2]} {q[3]}\n")
    print(f"Saved ground mesh: {out_path} ({len(pts)} vertices, {len(quads)} quads)")


def main():
    # Usage examples:
    #   Estimate point spacing only:
    #     python pc_preprocess.py input.las --estimate-spacing [--sample-size 5000]
    #
    #   Full pipeline (ground removal + noise removal):
    #     python pc_preprocess.py input.las --remove-ground
    #
    #   Ground removal only (skip noise removal):
    #     python pc_preprocess.py input.las --remove-ground --no-noise-removal
    #
    #   Noise removal only (skip ground removal):
    #     python pc_preprocess.py input.las
    #
    #   CSF parameters (for --remove-ground):
    #     --ground-resolution  ground mesh size in meters; smaller = more terrain detail, slower (default: 0.5)
    #     --class-threshold    distance threshold to classify as ground (default: 0.5)
    #     --rigidness          1=mountainous, 2=complex, 3=flat terrain (default: 3)
    #
    #   SOR parameters (for noise removal):
    #     --nn   number of neighbors (default: 10)
    #     --std  std ratio threshold; smaller = more aggressive (default: 1.0)
    #
    #   Output: <input_stem>_clean.<ext> saved in the same directory as input
    parser = argparse.ArgumentParser(description="Point cloud preprocessing: ground removal + noise removal")
    parser.add_argument("input", help="Input file (.las, .laz, or .ply)")

    # spacing estimate
    parser.add_argument("--estimate-spacing", action="store_true",
                        help="Estimate average point spacing and exit (no filtering)")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Number of points to sample for spacing estimate (default: 5000)")

    # ground removal
    parser.add_argument("--remove-ground", action="store_true",
                        help="Remove ground points using CSF before noise filtering")
    parser.add_argument("--ground-resolution", type=float, default=0.5,
                        help="CSF ground mesh resolution in meters (default: 0.5)")
    parser.add_argument("--class-threshold", type=float, default=0.2,
                        help="CSF distance threshold for ground classification (default: 0.2)")
    parser.add_argument("--rigidness", type=int, default=3, choices=[1, 2, 3],
                        help="CSF rigidness: 1=mountainous, 2=complex, 3=flat (default: 3)")
    parser.add_argument("--save-ground", action="store_true",
                        help="Export CSF ground mesh as OBJ alongside the output file")

    # noise removal
    parser.add_argument("--no-noise-removal", action="store_true",
                        help="Skip SOR noise removal")
    parser.add_argument("--nn", type=int, default=10,
                        help="SOR number of neighbors (default: 10)")
    parser.add_argument("--std", type=float, default=1.0,
                        help="SOR std ratio threshold (default: 1.0)")

    # downsampling
    parser.add_argument("--downsample", type=int, default=1, metavar="X",
                        help="Uniform downsample factor: keep every X-th point (default: 1, no downsampling)")

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Error: file not found: {in_path}")

    suffix = in_path.suffix.lower()
    ds_tag = f"_{args.downsample}x" if args.downsample > 1 else ""
    out_path = in_path.with_stem(in_path.stem + "_clean" + ds_tag)
    meta_dir = in_path.parent / f"{out_path.stem}_meta"
    meta_dir.mkdir(exist_ok=True)

    print(f"Reading: {in_path}")

    if suffix in (".las", ".laz"):
        pcd, las = load_las(in_path)
    elif suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(in_path))
        las = None
    else:
        sys.exit(f"Error: unsupported format '{suffix}', use .las/.laz/.ply")

    print(f"Loaded {len(pcd.points)} points")

    if args.estimate_spacing:
        estimate_spacing(pcd, args.sample_size)
        return

    # track surviving indices relative to original point cloud
    active_idx = list(range(len(pcd.points)))
    ground_idx = []
    noise_idx = []
    downsample_removed_idx = []

    if args.remove_ground:
        non_ground, ground_mesh = remove_ground_csf(
            pcd, args.ground_resolution, args.class_threshold, args.rigidness,
            export_cloth=args.save_ground,
        )
        non_ground_set = set(non_ground)
        ground_idx = [active_idx[i] for i in range(len(active_idx)) if i not in non_ground_set]
        active_idx = [active_idx[i] for i in non_ground]
        pcd = pcd.select_by_index(non_ground)
        if ground_mesh is not None:
            ground_path = meta_dir / f"{out_path.stem}_ground.obj"
            save_ground_obj(ground_mesh, ground_path)

    if not args.no_noise_removal:
        _, sor_ind = pcd.remove_statistical_outlier(nb_neighbors=args.nn, std_ratio=args.std)
        removed = len(pcd.points) - len(sor_ind)
        print(f"Noise removal (SOR): removed {removed}/{len(pcd.points)} points "
              f"({100*removed/len(pcd.points):.1f}%)")
        sor_set = set(sor_ind)
        noise_idx = [active_idx[i] for i in range(len(active_idx)) if i not in sor_set]
        active_idx = [active_idx[i] for i in sor_ind]
        pcd = pcd.select_by_index(sor_ind)

    print(f"Remaining: {len(pcd.points)} points")

    if args.downsample > 1:
        n = len(pcd.points)
        ds_idx = list(range(0, n, args.downsample))
        ds_set = set(ds_idx)
        downsample_removed_idx = [active_idx[i] for i in range(n) if i not in ds_set]
        active_idx = [active_idx[i] for i in ds_idx]
        pcd = pcd.select_by_index(ds_idx)
        print(f"Downsampling ({args.downsample}x): {n} -> {len(pcd.points)} points")

    params = {
        "input": str(in_path),
        "ground_removal": {
            "enabled": args.remove_ground,
            "ground_resolution": args.ground_resolution,
            "class_threshold": args.class_threshold,
            "rigidness": args.rigidness,
        } if args.remove_ground else {"enabled": False},
        "noise_removal": {
            "enabled": not args.no_noise_removal,
            "nn": args.nn,
            "std": args.std,
        } if not args.no_noise_removal else {"enabled": False},
        "downsample": args.downsample,
    }
    params_path = meta_dir / f"{out_path.stem}_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved: {params_path}")

    # Index map: maps original Gaussian indices into four categories for back-projection.
    #   survivor    — points that passed all filters and entered treeiso; treeiso output[i] → Gaussian[survivor[i]]
    #   ground      — removed by CSF ground filter
    #   noise       — removed by SOR noise filter
    #   downsampled — passed ground/noise filters but removed by uniform downsampling; assign label via KNN from survivors
    index_map_path = meta_dir / f"{out_path.stem}_index_map.npz"
    np.savez(
        index_map_path,
        survivor=np.array(active_idx, dtype=np.int32),
        ground=np.array(ground_idx, dtype=np.int32),
        noise=np.array(noise_idx, dtype=np.int32),
        downsampled=np.array(downsample_removed_idx, dtype=np.int32),
    )
    print(f"Saved: {index_map_path}")

    if suffix in (".las", ".laz"):
        save_las(las, active_idx, out_path)
        print(f"Saved: {out_path}")
    else:
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"Saved: {out_path}")
        laz_path = out_path.with_suffix(".laz")
        save_ply_as_laz(pcd, laz_path)
        print(f"Saved: {laz_path}")


if __name__ == "__main__":
    main()
