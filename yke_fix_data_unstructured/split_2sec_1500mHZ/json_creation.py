#This script should produce the JSON samples structured in the same way as the ones used by mit1003. Adapt especially BIN_SIZE_S to you desired sample length
#this will use all fixations that fall within that bin into one sample
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

OUT_TRAIN_JSON = "./yke_fix_data_unstructured/split_2sec_1500mHZ/ykedata_2s_fixations_train.json"
OUT_VAL_JSON   = "./yke_fix_data_unstructured/split_2sec_1500mHZ/ykedata_2s_fixations_validation.json"
OUT_TEST_JSON  = "./yke_fix_data_unstructured/split_2sec_1500mHZ/ykedata_2s_fixations_test.json"

BIN_SIZE_S = 2.0 #so duaration of the sample
MAX_TIME_S = 90.0

TEST_VIDEO = 7
VAL_RATIO = 0.15
RANDOM_SEED = 42

DEFAULT_WIDTH = 1904
DEFAULT_HEIGHT = 988

# -------------CONFFIG for saving frame names------------
FRAME_STARTS = [1, 51, 100]   # corresponds to [0001, 0051, 0100]
FRAME_STEP = 100             # adds per bin_idx
FRAME_PAD = 4                # 0001
FRAME_EXT = ".jpeg"          # or ".png" depending on your files


# Which timestamp decides the 2s bin
TIME_FOR_BINNING = "t_mid_s"   # or "t_begin_s", "t_end_s"

# Durations in ms like your example; set False for seconds
DURATION_IN_MS = True

# If needed it is possible to normalize coordinates to [0,1]
NORMALIZE_XY = False

# ==========================

def make_frame_names(video: int, bin_idx: int) -> list:
    """
    Returns list of 3 frame filenames like:
    video_{video}_{window}_frame_0001.jpeg, video_{video}_{window}_frame_0051.jpeg, ...
    with indices shifted by bin_idx * FRAME_STEP.
    """
    offset = bin_idx * FRAME_STEP
    indices = [s + offset for s in FRAME_STARTS]
    return [
        f"video_{video}_frame_{idx:0{FRAME_PAD}d}{FRAME_EXT}"
        for idx in indices
    ]

def format_window(bin_idx: int, bin_size_s: float) -> str:
    start = int(round(bin_idx * bin_size_s))
    end = int(round((bin_idx + 1) * bin_size_s))
    return f"{start:02d}-{end:02d}"


def make_sample(pp, video,bin_idx, X, Y, T, split):
    frames = make_frame_names(video=video, bin_idx=bin_idx)
    return {
        "name": frames,      # replace with real filename if needed
        "subject": int(pp),
        "X": X,
        "Y": Y,
        "T": T,
        "length": len(X),
        "split": split,                 # kept for compatibility; also separated into files
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
    df[TIME_FOR_BINNING] = df[TIME_FOR_BINNING].astype(float)

    # Keep valid time range
    df = df[(df[TIME_FOR_BINNING] >= 0.0) & (df[TIME_FOR_BINNING] < MAX_TIME_S)].copy()

    # 2-second bin index
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

    grouped = df.groupby(["pp", "video", "bin_idx"], sort=True) #S amples with zero fixations are never created,They are implicitly excluded

    for (pp, video, bin_idx), g in grouped:
        if NORMALIZE_XY:
            X = (g["x_fix"] / float(DEFAULT_WIDTH)).tolist()
            Y = (g["y_fix"] / float(DEFAULT_HEIGHT)).tolist()
        else:
            X = g["x_fix"].tolist()
            Y = g["y_fix"].tolist()

        if DURATION_IN_MS:
            T = (g["duration_s"] * 1000.0).tolist()
        else:
            T = g["duration_s"].tolist()

        # stats
        total_samples += 1

        num_fix = len(X)
        total_fixations += num_fix
        max_fixations_in_sample = max(max_fixations_in_sample, num_fix)
        fixations_per_sample.append(num_fix)

        if video == TEST_VIDEO:
            sample = make_sample(pp, video, bin_idx, X, Y, T, "test")
            test_samples.append(sample)
        else:
            split = "validation" if (pp, video, bin_idx) in val_keys else "train"
            sample = make_sample(pp, video, bin_idx, X, Y, T, split)
            if split == "validation":
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    # Stable ordering (nice for reproducibility)
    def sort_key(s):
        return (s["subject"], s["name"])  # you can also add bin info if you store it

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
    print((f"Total assigned samples: {len(test_samples)+len(train_samples)+len(val_samples)}"))
    print("=====================================")


if __name__ == "__main__":
    build_scandiff_jsons()
