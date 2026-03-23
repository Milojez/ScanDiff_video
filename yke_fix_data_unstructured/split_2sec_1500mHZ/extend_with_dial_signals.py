"""
Extends each sample in the train/val/test JSONs produced by json_creation_vid_gaze.py
with per-frame dial signal data (angle, speed) from the CSV files in dial_signals/movie_signal_csv/.

Frame-to-row mapping:
  CSV row with frame_number=N corresponds to frame file *_frame_NNNN.png.
  i.e. frame 0001 -> frame_number 1 -> time_end 0.02s.

For each sample, only the signal values at the frames listed in 'name' are included.
angle and speed lists are 1-to-1 with the sample's frame list.
"""

import csv
import json
import re
from pathlib import Path

# ==========================
# CONFIG
# ==========================

BASE_DIR = Path(__file__).parent

IN_JSONS = [
    BASE_DIR / "ykedata_2s_fix_vid_gaze_train.json",
    BASE_DIR / "ykedata_2s_fix_vid_gaze_validation.json",
    BASE_DIR / "ykedata_2s_fix_vid_gaze_test.json",
]

CSV_DIR = BASE_DIR / "dial_signals" / "movie_signal_csv"

OUT_SUFFIX = "_signal"  # appended before .json on output filenames
ROUND_DECIMALS = 4      # decimal places for angle and speed values; set to None to disable

# ==========================


def csv_path(movie_index: int) -> Path:
    return CSV_DIR / f"movie_{movie_index:02d}_dial_signals.csv"


def load_csv_signals(movie_index: int) -> dict:
    """
    Load a movie CSV into {frame_number: {signal_id: {"dial_position": int, "angle": float, "speed": float}}}.
    Column format: angle_dial_id_{sid}_pos_{pos} — sid at index 3, pos at index 5.
    """
    path = csv_path(movie_index)
    signals = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        # {signal_id: (angle_col, speed_col, dial_position)}
        dial_meta = {}
        for col in fieldnames:
            if col.startswith("angle_dial_id_"):
                parts = col.split("_")
                sid = int(parts[3])
                pos = int(parts[5])
                dial_meta[sid] = {"angle_col": col, "pos": pos}
        for col in fieldnames:
            if col.startswith("speed_dial_id_"):
                sid = int(col.split("_")[3])
                dial_meta[sid]["speed_col"] = col

        for row in reader:
            fn = int(row["frame_number"])
            signals[fn] = {
                sid: {
                    "dial_position": meta["pos"],
                    "angle": float(row[meta["angle_col"]]),
                    "speed": float(row[meta["speed_col"]]),
                }
                for sid, meta in dial_meta.items()
            }
    return signals


def extract_video_and_frame(frame_name: str):
    """
    Parse 'video_1_frame_0201.png' -> (video_index=1, frame_number=201).
    """
    m = re.match(r"video_(\d+)_frame_(\d+)", frame_name)
    if not m:
        raise ValueError(f"Cannot parse frame name: {frame_name!r}")
    return int(m.group(1)), int(m.group(2))


def extend_samples(samples: list, dial_cache: dict) -> list:
    extended = []
    for sample in samples:
        frame_names = sample["name"]

        video_idx = None
        frame_numbers = []
        for frame_name in frame_names:
            vid_idx, frame_num = extract_video_and_frame(frame_name)
            if video_idx is None:
                video_idx = vid_idx
            frame_numbers.append(frame_num)

        if video_idx not in dial_cache:
            dial_cache[video_idx] = load_csv_signals(video_idx)
        signals = dial_cache[video_idx]

        # Determine dial IDs from first frame entry
        signal_ids = sorted(signals[frame_numbers[0]].keys())

        def _round(v):
            return round(v, ROUND_DECIMALS) if ROUND_DECIMALS is not None else v

        dials_out = [
            {
                "signal_id": sid,
                "dial_position": signals[frame_numbers[0]][sid]["dial_position"],
                "angle": [_round(signals[fn][sid]["angle"]) for fn in frame_numbers],
                "speed": [_round(signals[fn][sid]["speed"]) for fn in frame_numbers],
            }
            for sid in signal_ids
        ]

        extended_sample = dict(sample)
        extended_sample["dials"] = dials_out
        extended.append(extended_sample)

    return extended


def main():
    dial_cache = {}

    for in_path in IN_JSONS:
        if not in_path.exists():
            print(f"Skipping (not found): {in_path}")
            continue

        with open(in_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        extended = extend_samples(samples, dial_cache)

        out_path = in_path.with_name(in_path.stem + OUT_SUFFIX + in_path.suffix)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(extended, f, indent=4)

        print(f"{in_path.name} -> {out_path.name}  ({len(extended)} samples)")

    print("Done.")


if __name__ == "__main__":
    main()
