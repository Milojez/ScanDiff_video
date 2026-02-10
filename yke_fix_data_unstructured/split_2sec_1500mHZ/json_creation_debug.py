#this code adapts the original json_creation.py to check how many samples from the csv fixations file actually are empty inside (making them useless samples due to possible errors)
#with this debug code you can see all samples affected by being empty. 
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

BIN_SIZE_S = 2.0
MAX_TIME_S = 90.0

TEST_VIDEO = 7
VAL_RATIO = 0.15
RANDOM_SEED = 42

DEFAULT_WIDTH = 1904
DEFAULT_HEIGHT = 988

TIME_FOR_BINNING = "t_mid_s"   # or "t_begin_s", "t_end_s"

DURATION_IN_MS = True
NORMALIZE_XY = False

# Where to save the missing-window inspection CSV
MISSING_WINDOWS_CSV = "./missing_windows_to_check.csv"

# ==========================


def format_window(bin_idx: int, bin_size_s: float) -> str:
    """Return window string like '00-02', '02-04', ..."""
    start = int(round(bin_idx * bin_size_s))
    end = int(round((bin_idx + 1) * bin_size_s))
    return f"{start:02d}-{end:02d}"


def make_sample(pp, video, bin_idx, X, Y, T, split):
    window = format_window(bin_idx, BIN_SIZE_S)
    return {
        "name": f"video_{video}_{window}",
        "subject": int(pp),
        "X": X,
        "Y": Y,
        "T": T,
        "length": len(X),
        "split": split,
        "height": int(DEFAULT_HEIGHT),
        "width": int(DEFAULT_WIDTH),
    }


def show_rows_around_window(df, pp, video, t_start, t_end, pad_s=0.5):
    lo = t_start - pad_s
    hi = t_end + pad_s
    sub = df[
        (df["pp"] == pp) &
        (df["video"] == video) &
        (df[TIME_FOR_BINNING] >= lo) &
        (df[TIME_FOR_BINNING] < hi)
    ].copy()

    print(f"\n--- Rows around pp={pp}, video={video}, window={t_start}-{t_end} (pad={pad_s}s) ---")
    if len(sub) == 0:
        print("No rows found in the padded range either.")
        return

    cols = ["pp", "video", TIME_FOR_BINNING, "t_begin_s", "t_end_s", "duration_s", "x_fix", "y_fix"]
    cols = [c for c in cols if c in sub.columns]  # be robust if a column is missing
    print(sub.sort_values(TIME_FOR_BINNING)[cols].to_string(index=False))


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

    # Drop broken rows (keeps JSON strict + prevents NaN)
    df = df.dropna(subset=["pp", "video", "duration_s", "x_fix", "y_fix", TIME_FOR_BINNING]).copy()

    # Keep valid time range
    df = df[(df[TIME_FOR_BINNING] >= 0.0) & (df[TIME_FOR_BINNING] < MAX_TIME_S)].copy()

    # 2-second bin index
    df["bin_idx"] = (df[TIME_FOR_BINNING] // BIN_SIZE_S).astype(int)

    # ==========================================================
    # DEBUG BLOCK: find missing (pp, video, bin) windows
    # ==========================================================
    pps = sorted(df["pp"].unique())
    videos = sorted(df["video"].unique())
    bins = list(range(int(MAX_TIME_S // BIN_SIZE_S)))  # for 90/2 -> 0..44

    existing = df[["pp", "video", "bin_idx"]].drop_duplicates()

    expected = pd.MultiIndex.from_product(
        [pps, videos, bins],
        names=["pp", "video", "bin_idx"]
    ).to_frame(index=False)

    missing = expected.merge(existing, on=["pp", "video", "bin_idx"], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])

    missing["t_start"] = missing["bin_idx"] * BIN_SIZE_S
    missing["t_end"] = (missing["bin_idx"] + 1) * BIN_SIZE_S

    print("----- MISSING WINDOWS (EMPTY SAMPLES) -----")
    print("Expected total windows:", len(expected))
    print("Present windows (non-empty):", len(existing))
    print("Missing windows (empty):", len(missing))

    print("\nExamples of missing windows (first 20):")
    print(missing.head(40).to_string(index=False))

    Path(MISSING_WINDOWS_CSV).parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(MISSING_WINDOWS_CSV, index=False)
    print(f"\nSaved full missing list to: {MISSING_WINDOWS_CSV}")

    # Optional: print raw rows around first few missing windows
    # (comment out if too noisy)
    for _, r in missing.head(5).iterrows():
        show_rows_around_window(df, int(r["pp"]), int(r["video"]), float(r["t_start"]), float(r["t_end"]))
    # ==========================================================

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
    # Build samples and split into lists + stats
    # --------------------------
    train_samples = []
    val_samples = []
    test_samples = []

    total_samples = 0
    total_fixations = 0
    max_fixations_in_sample = 0
    fixations_per_sample = []

    grouped = df.groupby(["pp", "video", "bin_idx"], sort=True)

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

    # Stable ordering
    def sort_key(s):
        return (s["subject"], s["name"])

    train_samples.sort(key=sort_key)
    val_samples.sort(key=sort_key)
    test_samples.sort(key=sort_key)

    # Write files (strict JSON)
    for path, data in [
        (OUT_TRAIN_JSON, train_samples),
        (OUT_VAL_JSON, val_samples),
        (OUT_TEST_JSON, test_samples),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, allow_nan=False)

    avg_fixations = total_fixations / total_samples if total_samples > 0 else 0.0
    std_fixations = float(np.std(fixations_per_sample)) if fixations_per_sample else 0.0

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total samples generated: {total_samples}")
    print(f"Max fixations in a sample: {max_fixations_in_sample}")
    print(f"Average fixations per sample: {avg_fixations:.2f}")
    print(f"Std dev of fixations/sample: {std_fixations:.2f}")
    print("-------------------------------------")
    print(f"Train samples:      {len(train_samples)} -> {OUT_TRAIN_JSON}")
    print(f"Validation samples: {len(val_samples)} -> {OUT_VAL_JSON}")
    print(f"Test samples:       {len(test_samples)} -> {OUT_TEST_JSON}")
    print("=====================================")


if __name__ == "__main__":
    build_scandiff_jsons()

