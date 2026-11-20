"""
resume_net_proc.py

Continues paralel_net_proc.py after the power failure, WITHOUT reprocessing 2022 -> last saved day.

How it works
------------
* Every processing step (STA/LTA, Q-transform, matrix, PDFs, next-day overlap) is imported from
  paralel_net_proc.py, so the resumed run uses exactly the same parameters. Keep both files in the same folder.
* For each hw, day-files before the resume date are skipped without even being read from disk.
* The resume date itself is processed again from its first trigger. Reason: the script only writes a
  parquet part every 200 triggers, so the rows that were still in memory when the power went off
  (the end of the last day, possibly part of the next one) were never saved.
* Everything new goes to a NEW folder per hw (<old folder>_resume). The old folders are never touched.
* Because the resume day exists in both places, use load_full_dataset() (bottom of this file) to read
  old + new parquets together: it drops the old rows from the resume date onwards, so there are no duplicates.

Run it by hand, like the original:  python3 resume_net_proc.py
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from obspy import read

from paralel_net_proc import (
    PATH, YEARS, STATIONS, CHANNELS,
    FRANGE, QRANGE, WHITEN, HALF_WIDTH, MATRIX_HALF_WIDTHS, MAX_WORKERS,
    STA, LTA, ON_THRESHOLD, OFF_THRESHOLD,
    FREQ_BINS, TIME_BINS, INTENSITY_THRESHOLD, PARQUET_CHUNK_SIZE, OUTPUT_PREFIX,
    seismic_trig, generate_qtransform, qtransform_to_matrix, get_q_value, plot_qtransform_pdf,
    parse_date_from_filename, write_parquet_part, find_next_day_file, extend_with_next_day, window_tag,
)

#######################################################################
# Where each window stopped (DD-MM-YYYY = last day that has rows in the parquet parts)
#######################################################################

RESUME_FROM = {
    1.0:  "28-11-2025",   # 739 parts
    2.0:  "25-09-2025",   # 715 parts
    4.0:  "27-10-2025",   # 730 parts
    6.0:  "15-10-2025",   # 726 parts
    12.0: "11-11-2025",   # 732 parts
}

INPUT_DATE_FORMAT = "%d-%m-%Y"


def old_output_dir(matrix_half_width):
    """
    Folder created by the crashed run. make_output_dir() adds _1, _2, ... when a folder with that name already
    existed, so if your crashed run ended up in e.g. '..._ont15_1', change the name here.
    """
    return Path(f"{OUTPUT_PREFIX}_hw{window_tag(matrix_half_width)}_ont{ON_THRESHOLD}")


#######################################################################
# Sanity checks on the crashed run
#######################################################################

def check_old_run(matrix_half_width, old_dir, user_date):
    """
    Opens the LAST parquet part of the crashed run and reads the last date stored in it.
    A power cut can leave the last file truncated/corrupted, and a typo in RESUME_FROM would silently skip data,
    so this is verified before anything starts. Returns the date to resume from: the earlier of the date you
    typed and the date actually found (reprocessing a few extra triggers is harmless, skipping some is not).
    """
    tag = window_tag(matrix_half_width)
    parts = sorted((old_dir / "parquet_parts").glob("qtransform_features_part_*.parquet"))
    if not parts:
        raise SystemExit(f"[hw={tag}] no parquet parts found in {old_dir / 'parquet_parts'}")

    try:
        last_saved = pd.read_parquet(parts[-1], columns=["date"])["date"].max()
    except Exception as e:
        raise SystemExit(
            f"[hw={tag}] the last part ({parts[-1].name}) cannot be read, probably cut by the power failure: {e}\n"
            f"Delete (or move away) that file and run this script again: it will then resume from the part before it.")

    found_date = datetime.strptime(last_saved, "%Y-%m-%d")
    if found_date != user_date:
        print(f"[hw={tag}] WARNING: RESUME_FROM says {user_date:%Y-%m-%d} but the last part has {found_date:%Y-%m-%d}. "
              f"Using the earlier one.", flush=True)

    start = min(found_date, user_date)
    print(f"[hw={tag}] old run: {len(parts)} parts, last saved day {found_date:%Y-%m-%d} -> resuming from {start:%Y-%m-%d}", flush=True)
    return start


def make_resume_dir(old_dir):
    """<old folder>_resume (adds _1, _2, ... if it already exists, so a previous resume is never overwritten)."""
    base_dir = Path(f"{old_dir}_resume")
    output_dir = base_dir
    counter = 1
    while output_dir.exists():
        output_dir = Path(f"{base_dir}_{counter}")
        counter += 1
    output_dir.mkdir()
    return output_dir


#######################################################################
# Worker: same as process_window() of the original, but skipping everything before start_date
#######################################################################

def process_window_resume(matrix_half_width, old_dir, output_dir, start_date):
    tag = window_tag(matrix_half_width)

    def log(message):
        print(f"[hw={tag}] {message}", flush=True)

    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"RESUMED RUN: starts at {start_date:%Y-%m-%d}; earlier days are in {old_dir}\n")
        f.write(f"Inputs: FRANGE = {FRANGE}, QRANGE = {QRANGE}, WHITEN = {WHITEN}, HALF_WIDTH = {HALF_WIDTH}, MATRIX_HALF_WIDTH = {matrix_half_width}\n")
        f.write(f"Parameters: sta = {STA}, lta = {LTA}, on_threshold = {ON_THRESHOLD}, off_threshold = {OFF_THRESHOLD}\n")
        f.write(f"Images: frequency_bins = {FREQ_BINS}, time_bins = {TIME_BINS}, intensity_threshold = {INTENSITY_THRESHOLD}\n")

    parts_dir = output_dir / "parquet_parts"
    parts_dir.mkdir()
    part_idx = 0

    rows = []
    total_files = 0
    skipped_files = 0
    failed_files = []
    total_triggers = 0
    successful_triggers = 0
    failed_triggers = 0

    for year in YEARS:
        if int(year) < start_date.year:
            continue   # whole year already done

        run_dir = PATH / year

        if not run_dir.exists():
            log(f"Run folder not found, skipping: {run_dir}")
            continue

        for station in STATIONS:
            for channel in CHANNELS:

                pattern = f"{station}_{channel}_*.mseed"
                files = sorted(run_dir.glob(pattern))

                if not files:
                    log(f"No files found for {year}/{station}_{channel}")
                    continue

                for mseed_file in files:
                    file_date = parse_date_from_filename(mseed_file.name)

                    # RESUME: days before start_date were already processed -> skip without reading the file
                    if file_date is not None and file_date < start_date:
                        skipped_files += 1
                        continue

                    total_files += 1
                    log(f"Processing: {mseed_file}")

                    try:
                        st = read(mseed_file, format="mseed")
                        tr = st[0]
                        starttime = tr.stats.starttime
                        endtime = tr.stats.endtime
                    except Exception as e:
                        log(f"Failed to read {mseed_file}: {e}")
                        failed_files.append(str(mseed_file))
                        continue

                    if file_date is None:
                        log(f"Could not parse a date from filename, skipping: {mseed_file.name}")
                        failed_files.append(str(mseed_file))
                        continue

                    triggers = seismic_trig(mseed_file, FRANGE[0], FRANGE[1], UTC=True, p=False)
                    log(f"Number of triggers: {len(triggers)}")

                    if len(triggers) == 0:
                        log("No triggers found.")
                        continue

                    # Triggers closer than HALF_WIDTH to the end of the file borrow the first seconds of the
                    # next day's file (same logic as the original script).
                    tr_qt = tr
                    n_near_end = sum(1 for t in triggers if t + HALF_WIDTH > endtime)
                    if n_near_end > 0:
                        next_file = find_next_day_file(file_date, station, channel)
                        if next_file is None:
                            log(f"{n_near_end} trigger(s) near the end of the file, but no next-day file was found: windows stay truncated")
                        else:
                            tr_ext = extend_with_next_day(tr, next_file)
                            if tr_ext is None:
                                log(f"{n_near_end} trigger(s) near the end of the file, but {next_file.name} is empty/unreadable/not contiguous: windows stay truncated")
                            else:
                                tr_qt = tr_ext
                                log(f"{n_near_end} trigger(s) near the end of the file: windows extended with {next_file.name}")

                    total_triggers += len(triggers)
                    month_year = file_date.strftime("%b_%Y").lower()
                    day_str = file_date.strftime("%Y%m%d")
                    file_out_dir = output_dir / year / station / channel / month_year
                    file_out_dir.mkdir(parents=True, exist_ok=True)

                    for i, trigger_time in enumerate(triggers):
                        try:
                            qspec = generate_qtransform(tr_qt, trigger_time, HALF_WIDTH)
                            matrix = qtransform_to_matrix(qspec, interval=matrix_half_width, nt=TIME_BINS, nf=FREQ_BINS, frange=FRANGE, intensity_threshold=INTENSITY_THRESHOLD)

                            expected_shape = (FREQ_BINS, TIME_BINS)
                            if matrix.shape != expected_shape:
                                raise ValueError(f"Unexpected matrix shape: {matrix.shape}. Expected {expected_shape}.")

                            qplot_file = file_out_dir / f"qtransform_{day_str}_{i:06d}.pdf"
                            plot_qtransform_pdf(qspec=qspec, matrix=matrix, trigger_time=trigger_time, center_time=trigger_time, half_width=matrix_half_width, output_file=qplot_file, intensity_threshold=INTENSITY_THRESHOLD)

                            features = matrix.flatten()
                            row = {"year": year, "station": station, "channel": channel, "file": mseed_file.name, "date": file_date.strftime("%Y-%m-%d"), "month_year": month_year, "trigger_time": str(trigger_time), "q_value": get_q_value(qspec)}
                            for j, value in enumerate(features):
                                row[f"feature_{j:04d}"] = value

                            rows.append(row)
                            successful_triggers += 1
                            log(f"Matrix: {matrix.shape} | Features: {features.shape}")

                        except Exception as e:
                            failed_triggers += 1
                            log(f"Failed trigger {i + 1} ({trigger_time}): {e}")

                        if len(rows) >= PARQUET_CHUNK_SIZE:
                            part_file = write_parquet_part(rows, parts_dir, part_idx)
                            log(f"Parquet part saved: {part_file} ({len(rows)} triggers)")
                            part_idx += 1
                            rows.clear()

    log(f"Skipped {skipped_files} day-files before {start_date:%Y-%m-%d}")

    result = {
        "window": matrix_half_width,
        "output_dir": str(output_dir),
        "total_files": total_files,
        "failed_files": len(failed_files),
        "total_triggers": total_triggers,
        "successful_triggers": successful_triggers,
        "failed_triggers": failed_triggers,
        "parquet_parts": part_idx}

    log(f"Files: {total_files} (failed: {len(failed_files)}) | Triggers: {total_triggers} | Successful: {successful_triggers} | Failed: {failed_triggers}")

    if successful_triggers == 0:
        log("No Q-transforms were successfully processed.")
        return result

    # Flush the last, partial chunk
    if rows:
        part_file = write_parquet_part(rows, parts_dir, part_idx)
        log(f"Parquet part saved: {part_file} ({len(rows)} triggers)")
        part_idx += 1
        rows.clear()
    result["parquet_parts"] = part_idx

    n_columns = 8 + FREQ_BINS * TIME_BINS
    log(f"Total rows: {successful_triggers} in {part_idx} parquet part(s) at {parts_dir}")

    with open(summary_file, "a") as f:
        f.write(f"\nTotal files processed: {total_files} | Failed files: {len(failed_files)} | Skipped (before resume date): {skipped_files}\n")
        f.write(f"Total triggers: {total_triggers} | Successful triggers: {successful_triggers} | Failed triggers: {failed_triggers}\n")
        f.write(f"Parquet: {successful_triggers} rows x {n_columns} columns in {part_idx} part(s) of up to {PARQUET_CHUNK_SIZE} rows, at {parts_dir}\n")
        if failed_files:
            f.write("Failed files:\n")
            for ff in failed_files:
                f.write(f"  {ff}\n")

    log(f"Summary saved to: {summary_file}")
    return result


#######################################################################
# Reading old + new together
#######################################################################

def load_full_dataset(resume_dir):
    """
    Returns ONE dataframe with the whole dataset of a window: the old parquet parts (only days BEFORE the resume
    date, so the partially-saved last day is not duplicated) + the new parquet parts.

        df = load_full_dataset("TERZ_spectrograms_matrix_results_cut_hw12_ont15_resume")
    """
    resume_dir = Path(resume_dir)
    info = json.loads((resume_dir / "resume_info.json").read_text())

    old = ds.dataset(Path(info["old_dir"]) / "parquet_parts", format="parquet")
    old_df = old.to_table(filter=ds.field("date") < info["resume_from"]).to_pandas()

    new = ds.dataset(resume_dir / "parquet_parts", format="parquet")
    new_df = new.to_table().to_pandas()

    return pd.concat([old_df, new_df], ignore_index=True)


#######################################################################
# Main: one process per window (same as the original)
#######################################################################

if __name__ == "__main__":
    windows = list(MATRIX_HALF_WIDTHS)

    missing = [hw for hw in windows if hw not in RESUME_FROM]
    if missing:
        raise SystemExit(f"RESUME_FROM has no date for these windows: {missing}")

    # 1) Check every crashed run first, so nothing is created if something is wrong
    start_dates = {}
    old_dirs = {}
    for hw in windows:
        old_dirs[hw] = old_output_dir(hw)
        if not old_dirs[hw].exists():
            raise SystemExit(f"Old output folder not found: {old_dirs[hw]} (edit old_output_dir())")
        start_dates[hw] = check_old_run(hw, old_dirs[hw], datetime.strptime(RESUME_FROM[hw], INPUT_DATE_FORMAT))

    # 2) New folders, created one after the other before any process starts
    output_dirs = {}
    for hw in windows:
        output_dirs[hw] = make_resume_dir(old_dirs[hw])
        info = {"old_dir": str(old_dirs[hw]), "resume_from": start_dates[hw].strftime("%Y-%m-%d")}
        (output_dirs[hw] / "resume_info.json").write_text(json.dumps(info, indent=2))
        print(f"Pasta criada: {output_dirs[hw]}")

    results = []
    failed_windows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_window_resume, hw, old_dirs[hw], output_dirs[hw], start_dates[hw]): hw for hw in windows}

        for future in as_completed(futures):
            hw = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                failed_windows.append(hw)
                print(f"[hw={window_tag(hw)}] WINDOW FAILED with {type(e).__name__}: {e}", flush=True)

    print()
    print("=" * 70)
    print("RESUMED DATASET (new folders only; use load_full_dataset() to read old + new)")
    print("=" * 70)
    for r in sorted(results, key=lambda r: r["window"]):
        print(f"hw = {r['window']:g} s | files {r['total_files']} (failed {r['failed_files']}) | "
              f"triggers {r['total_triggers']} | ok {r['successful_triggers']} | failed {r['failed_triggers']} | "
              f"{r['parquet_parts']} parquet part(s) | {r['output_dir']}")

    if failed_windows:
        print(f"\nWindows that crashed (see the messages above): {sorted(failed_windows)}")
        raise SystemExit(1)
