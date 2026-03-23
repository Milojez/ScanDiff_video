"""
This script takes the json file for dial signals produced by matlab and makes them nice csv files
Creates a CSV for each movie dial signals JSON.
Columns: frame_number, time_end, angle_dial_id_{signal_id}_pos_{dial_position}, speed_dial_id_{signal_id}_pos_{dial_position}, ...

Dials are ordered by signal_id (1..6).
frame N ends at time N * (1 / sampling_rate_hz).
"""

import csv
import json
from pathlib import Path

DIAL_SIGNALS_DIR = Path(__file__).parent / "dial_signals"
OUT_DIR = DIAL_SIGNALS_DIR / "movie_signal_csv"   # save CSVs alongside the JSONs # save CSVs alongside the JSONs


def make_csv(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sampling_rate = data["sampling_rate_hz"]
    frame_dt = 1.0 / sampling_rate  # seconds per frame

    # Sort dials by signal_id so columns are dial_1 ... dial_6
    dials = sorted(data["dials"], key=lambda d: d["signal_id"])
    num_dials = len(dials)
    num_frames = data["num_frames"]

    # dial_order maps position number (str) -> position name e.g. "1" -> "top_left"
    # Build header
    header = ["frame_number", "time_end"]
    for d in dials:
        sid = d["signal_id"]
        pos = d["dial_position"]
        header += [f"angle_dial_id_{sid}_pos_{pos}", f"speed_dial_id_{sid}_pos_{pos}"]

    out_path = OUT_DIR / json_path.with_suffix(".csv").name
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(num_frames):
            frame_number = i + 1
            time_end = round(frame_number * frame_dt, 6)
            row = [frame_number, time_end]
            for d in dials:
                row.append(d["relative_angle"][i])
                row.append(d["velocity"][i])
            writer.writerow(row)

    return out_path, num_frames


def main():
    json_files = sorted(DIAL_SIGNALS_DIR.glob("movie_*_dial_signals.json"))
    if not json_files:
        print(f"No dial signal JSONs found in {DIAL_SIGNALS_DIR}")
        return

    for json_path in json_files:
        out_path, n = make_csv(json_path)
        print(f"{json_path.name} -> {out_path.name}  ({n} frames)")

    print("Done.")


if __name__ == "__main__":
    main()
