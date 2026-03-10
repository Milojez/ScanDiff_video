#This script should produce the JSON samples structured in the same way as the ones used by mit1003. Adapt especially BIN_SIZE_S to you desired sample length
#this will use all fixations that fall within that bin into one sample
#what is important this script already expects the fixation being splitted when necessary

#basically this code is an update dversion of json_creation.py where here  is also entry delta_t_start added for each sample
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================
# USER CONFIG
# ==========================

CSV_FILES = [
    "./yke_fix_data_unstructured/split_2sec_1500mHZ/GT_all_fixation_noDial7_noPPs_split_2s.csv",
]

OUT_TRAIN_JSON = "./yke_fix_data_unstructured/split_2sec_1500mHZ/ykedata_2s_fix_vid_gaze_train.json"
OUT_VAL_JSON   = "./yke_fix_data_unstructured/split_2sec_1500mHZ/ykedata_2s_fix_vid_gaze_validation.json"
OUT_TEST_JSON  = "./yke_fix_data_unstructured/split_2sec_1500mHZ/ykedata_2s_fix_vid_gaze_test.json"

BIN_SIZE_S = 2.0   # duration of one sample (already inspired by how the fixations were splitted at boundaries( 2s, 3s etc.) in the loaded file)
MAX_TIME_S = 90.0

TEST_VIDEO = 7
VAL_RATIO = 0.15
RANDOM_SEED = 42

DEFAULT_WIDTH = 1904
DEFAULT_HEIGHT = 988

# ------------- CONFIG for saving frame names ------------
FRAME_STARTS = [1, 51, 100]
FRAME_STEP = 100
FRAME_PAD = 4
FRAME_EXT = ".png"

# Which timestamp decides the bin
TIME_FOR_BINNING = "t_mid_s"   # or "t_begin_s", "t_end_s"

# Save durations in ms like in your example
DURATION_IN_MS = True

# Normalize coordinates to [0,1] if needed
NORMALIZE_XY = False

# ==========================

def make_frame_names(video: int, bin_idx: int) -> list:
    video = int(video)
    offset = int(bin_idx * FRAME_STEP)
    indices = [int(s + offset) for s in FRAME_STARTS]
    return [
        f"video_{video}_frame_{idx:0{FRAME_PAD}d}{FRAME_EXT}"
        for idx in indices
    ]


def format_window(bin_idx: int, bin_size_s: float) -> str:
    start = int(round(bin_idx * bin_size_s))
    end = int(round((bin_idx + 1) * bin_size_s))
    return f"{start:02d}-{end:02d}"


def make_sample(pp, video, bin_idx, X, Y, delta_t_start, duration, split):
    frames = make_frame_names(video=video, bin_idx=bin_idx)
    return {
        "name": frames,
        "subject": int(pp),
        "X": X,
        "Y": Y,
        "delta_t_start": delta_t_start,
        "T": duration,
        "length": len(X),
        "split": split,
        "height": int(DEFAULT_HEIGHT),
        "width": int(DEFAULT_WIDTH),
    }


def build_scandiff_jsons():
    # Load CSV(s)
    dfs = [pd.read_csv(f) for f in CSV_FILES]
    df = pd.concat(dfs, ignore_index=True)

    # Types
    df["pp"] = df["pp"].astype(int)
    df["video"] = df["video"].astype(int)
    df["duration_s"] = df["duration_s"].astype(float)
    df["x_fix"] = df["x_fix"].astype(float)
    df["y_fix"] = df["y_fix"].astype(float)
    df["t_begin_s"] = df["t_begin_s"].astype(float)
    df["t_end_s"] = df["t_end_s"].astype(float)
    df[TIME_FOR_BINNING] = df[TIME_FOR_BINNING].astype(float)

    # Keep valid time range
    df = df[(df[TIME_FOR_BINNING] >= 0.0) & (df[TIME_FOR_BINNING] < MAX_TIME_S)].copy()

    # Bin index
    df["bin_idx"] = (df[TIME_FOR_BINNING] // BIN_SIZE_S).astype(int)

    # --------------------------------------------------
    # Expected number of samples (theoretical maximum)
    # --------------------------------------------------
    num_bins = int(MAX_TIME_S // BIN_SIZE_S)

    num_pps = df["pp"].nunique()
    num_videos = df["video"].nunique()

    expected_samples = num_pps * num_videos * num_bins

    # --------------------------
    # Determine validation keys (only for non-test videos)
    # --------------------------
    non_test_keys = (
        df[df["video"] != TEST_VIDEO][["pp", "video", "bin_idx"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    non_test_keys = list(non_test_keys)

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(non_test_keys)

    n_val = int(round(len(non_test_keys) * VAL_RATIO))
    val_keys = set(non_test_keys[:n_val])

    # --------------------------
    # Build samples and split into lists
    # --------------------------
    train_samples = []
    val_samples = []
    test_samples = []

    total_samples = 0
    max_fixations_in_sample = 0
    total_fixations = 0
    fixations_per_sample = []

    grouped = df.groupby(["pp", "video", "bin_idx"], sort=True)

    for (pp, video, bin_idx), g in grouped:
        # Make sure fixations inside one sample are in temporal order
        g = g.sort_values("t_begin_s").reset_index(drop=True)

        if NORMALIZE_XY:
            X = (g["x_fix"] / float(DEFAULT_WIDTH)).tolist()
            Y = (g["y_fix"] / float(DEFAULT_HEIGHT)).tolist()
        else:
            X = g["x_fix"].tolist()
            Y = g["y_fix"].tolist()

        # duration
        if DURATION_IN_MS:
            duration = [round(v, 2) for v in (g["duration_s"] * 1000.0).tolist()]
        else:
            duration = [round(v, 2) for v in g["duration_s"].tolist()]

        # delta_t_start:
        # first fixation relative to sample start
        # next fixations relative to previous fixation end
        sample_start = bin_idx * BIN_SIZE_S
        delta_t_start = []

        prev_end = sample_start
        for _, row in g.iterrows():
            gap = row["t_begin_s"] - prev_end
            value = gap * 1000.0 if DURATION_IN_MS else gap
            delta_t_start.append(round(value, 2))
            prev_end = row["t_end_s"]

        # stats
        total_samples += 1

        num_fix = len(X)
        total_fixations += num_fix
        max_fixations_in_sample = max(max_fixations_in_sample, num_fix)
        fixations_per_sample.append(num_fix)

        if video == TEST_VIDEO:
            sample = make_sample(pp, video, bin_idx, X, Y, delta_t_start, duration, "test")
            test_samples.append(sample)
        else:
            split = "validation" if (pp, video, bin_idx) in val_keys else "train"
            sample = make_sample(pp, video, bin_idx, X, Y, delta_t_start, duration, split)
            if split == "validation":
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    # Stable ordering
    def sort_key(s):
        return (s["subject"], s["name"])

    train_samples.sort(key=sort_key)
    val_samples.sort(key=sort_key)
    test_samples.sort(key=sort_key)

    # Write files
    for path, data in [
        (OUT_TRAIN_JSON, train_samples),
        (OUT_VAL_JSON, val_samples),
        (OUT_TEST_JSON, test_samples),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # Print summary stats
    print("========== DATASET SUMMARY ==========")
    avg_fixations = total_fixations / total_samples if total_samples > 0 else 0.0
    std_fixations = float(np.std(fixations_per_sample)) if fixations_per_sample else 0.0

    skipped_zero_fix_samples = expected_samples - total_samples

    print("-------------------------------------")
    print(f"Expected samples (pp × video × bins): {expected_samples}")
    print(f"Actual samples generated:             {total_samples}")
    print(f"Skipped zero-fixation samples:        {skipped_zero_fix_samples}")
    print("Zero-fixation samples are excluded implicitly by groupby from produced jsons.")

    print("-------------------------------------")
    print(f"Average fixations per sample: {avg_fixations:.2f}")
    print(f"Std dev of fixations/sample: {std_fixations:.2f}")
    print(f"Max fixations in a sample: {max_fixations_in_sample}")
    print("-------------------------------------")
    print(f"Train samples:      {len(train_samples)} -> {OUT_TRAIN_JSON}")
    print(f"Validation samples: {len(val_samples)} -> {OUT_VAL_JSON}")
    print(f"Test samples:       {len(test_samples)} -> {OUT_TEST_JSON}")
    print(f"Total assigned samples: {len(test_samples) + len(train_samples) + len(val_samples)}")
    print("=====================================")


if __name__ == "__main__":
    build_scandiff_jsons()