"""Tree isolation from terrestrial laser scanning point clouds."""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from glob import glob

import laspy
import numpy as np
import pandas as pd

import treeiso_core as core


def shuffle_labels(labels):
    """Randomly permute label IDs while preserving which points belong to the same group."""
    unique_ids = np.unique(labels)
    shuffled_ids = unique_ids.copy()
    np.random.shuffle(shuffled_ids)
    mapping = np.zeros(unique_ids.max() + 1, dtype=np.int32)
    mapping[unique_ids] = shuffled_ids
    return mapping[labels]


def save_params_json(out_dir, if_isolate_outlier, if_shuffle_labels):
    params = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": {
            "PR_REG_STRENGTH1": core.PR_REG_STRENGTH1,
            "PR_MIN_NN1": core.PR_MIN_NN1,
            "PR_REG_STRENGTH2": core.PR_REG_STRENGTH2,
            "PR_MIN_NN2": core.PR_MIN_NN2,
            "PR_DECIMATE_RES1": core.PR_DECIMATE_RES1,
            "PR_DECIMATE_RES2": core.PR_DECIMATE_RES2,
            "PR_MAX_GAP": core.PR_MAX_GAP,
            "PR_REL_HEIGHT_LENGTH_RATIO": core.PR_REL_HEIGHT_LENGTH_RATIO,
            "PR_VERTICAL_WEIGHT": core.PR_VERTICAL_WEIGHT,
            "PR_MIN_NN3": core.PR_MIN_NN3,
            "PR_SCORE_CANDIDATE_THRESH": core.PR_SCORE_CANDIDATE_THRESH,
            "PR_INIT_STEM_REL_LENGTH_THRESH": core.PR_INIT_STEM_REL_LENGTH_THRESH,
            "PR_MAX_OUTLIER_GAP": core.PR_MAX_OUTLIER_GAP,
        },
        "options": {
            "isolate_outlier": if_isolate_outlier,
            "shuffle_labels": if_shuffle_labels,
        },
    }
    json_path = out_dir / "params.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved: {json_path}")


def process_las_file(path_to_las, if_isolate_outlier=False, if_shuffle_labels=False):
    """Process a LAS/LAZ file."""
    in_path = Path(path_to_las)
    out_dir = in_path.parent / f"{in_path.stem}_treeiso_output"
    try:
        out_dir.mkdir(exist_ok=False)
    except FileExistsError:
        sys.exit(f"Error: output folder already exists: {out_dir}\nDelete or rename it before rerunning.")

    print('*******Processing LAS/LAZ******* ' + path_to_las)
    las = laspy.read(path_to_las)

    # Extract point cloud
    pcd = np.transpose([las.x, las.y, las.z])

    # Process the point cloud
    init_labels, intermediate_labels, final_labels, dec_inverse_idx, dec_inverse_idx2 = core.process_point_cloud(pcd)

    if if_isolate_outlier:
        connected_labels = core.isolate_gaps(pcd, core.PR_MAX_OUTLIER_GAP)
        _, final_labels = np.unique(np.transpose([final_labels, connected_labels[dec_inverse_idx]]),
                                    axis=0, return_inverse=True)

    print(f"Segments: init={len(np.unique(init_labels))}  intermediate={len(np.unique(intermediate_labels))}  final={len(np.unique(final_labels))}")

    # Add labels to LAS file (non-shuffled)
    try:
        las.add_extra_dim(laspy.ExtraBytesParams(name="init_segs", type="int32", description="init_segs"))
        las.init_segs = init_labels[dec_inverse_idx]

        las.add_extra_dim(laspy.ExtraBytesParams(name="intermediate_segs", type="int32", description="intermediate_segs"))
        las.intermediate_segs = intermediate_labels[dec_inverse_idx2]

        las.add_extra_dim(laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs"))
        las.final_segs = final_labels[dec_inverse_idx]
    except ValueError as e:
        sys.exit(f"Error: {path_to_las} already contains treeiso label fields. Please use the original (unprocessed) file.\nDetail: {e}")

    out_path = out_dir / f"{in_path.stem}_treeiso.laz"
    las.write(str(out_path))
    print(f"Saved: {out_path}")

    if if_shuffle_labels:
        las.init_segs = shuffle_labels(init_labels)[dec_inverse_idx]
        las.intermediate_segs = shuffle_labels(intermediate_labels)[dec_inverse_idx2]
        las.final_segs = shuffle_labels(final_labels)[dec_inverse_idx]
        shuffle_path = out_dir / f"{in_path.stem}_treeiso_shuffle.laz"
        las.write(str(shuffle_path))
        print(f"Saved: {shuffle_path}")

    save_params_json(out_dir, if_isolate_outlier, if_shuffle_labels)

    print('*******End processing*******')


def read_csv_file(path_to_csv):
    first_line = ""
    with open(path_to_csv, 'r') as f:
        first_line = f.readline().strip()
    tokens = first_line.strip().split()
    if not tokens:
        return None
    if (tokens[0].lstrip('-').replace('.', '', 1)).isnumeric():
        df = pd.read_csv(path_to_csv, header=None, sep=' |;|,|\\t')
        pcd = df.to_numpy()[:, :3]
        return df, pcd, False
    else:
        if first_line.startswith('//') or first_line.startswith('#'):
            first_line = first_line.lstrip('/#').strip()

        try:
            df = pd.read_csv(path_to_csv, header=0, sep=' |;|,|\\t')
        except Exception as e:
            print(f"Csv header parsing failed: {e}")
            return None, None, None

        column_names = first_line.split()
        column_names_lower = [col.lower() for col in column_names]
        if 'x' in column_names_lower and 'y' in column_names_lower and 'z' in column_names_lower:
            x_idx = column_names_lower.index('x')
            y_idx = column_names_lower.index('y')
            z_idx = column_names_lower.index('z')
            pcd = np.array(df.iloc[:, [x_idx, y_idx, z_idx]])
            return df, pcd, True


def process_csv_file(path_to_csv):
    """Process a CSV/TXT file."""
    print('*******Processing CSV/TXT******* ' + path_to_csv)

    df, pcd, has_header = read_csv_file(path_to_csv)
    if df is None:
        print(f"Unable to process {path_to_csv}. Skipping...")
        return

    init_labels, intermediate_labels, final_labels, dec_inverse_idx, dec_inverse_idx2 = core.process_point_cloud(pcd)

    if has_header:
        output_df = df.copy()
    else:
        output_df = pd.DataFrame(df.values, columns=[f'col{i + 1}' for i in range(df.shape[1])])

    output_df['init_segs'] = init_labels[dec_inverse_idx]
    output_df['intermediate_segs'] = intermediate_labels[dec_inverse_idx2]
    output_df['final_segs'] = final_labels[dec_inverse_idx]

    output_path = path_to_csv[:-4] + "_treeiso" + os.path.splitext(path_to_csv)[1]
    if has_header:
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_csv(output_path, index=False, header=False)

    print('*******End processing*******')


def main():
    """Main function to process laser scanning point clouds."""
    parser = argparse.ArgumentParser(description="Individual-tree isolation (treeiso) from TLS point clouds")
    parser.add_argument("path", help="Directory containing .las/.laz or .csv files")
    parser.add_argument("--isolate-outlier", action="store_true", help="Post-process to remove isolated outlier points")
    parser.add_argument("--shuffle-labels", action="store_true", help="Shuffle label IDs for better color contrast in visualization")
    args = parser.parse_args()

    path_input = args.path

    # Process LAS/LAZ files
    pathes_to_las = glob(os.path.join(path_input, "*.la[sz]"))
    for path_to_las in pathes_to_las:
        process_las_file(path_to_las, if_isolate_outlier=args.isolate_outlier, if_shuffle_labels=args.shuffle_labels)

    # Process CSV/TXT files
    pathes_to_csv = glob(os.path.join(path_input, "*.csv"))
    for path_to_csv in pathes_to_csv:
        process_csv_file(path_to_csv)

    if len(pathes_to_las) == 0 and len(pathes_to_csv) == 0:
        print('Failed to find the las/laz or csv/txt files from your input directory')
        return


if __name__ == '__main__':
    main()
