import matplotlib
matplotlib.use("Agg")  # headless/multiprocess-safe backend - must be set before pyplot import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import gc
import logging
import time
import traceback
from datetime import datetime
from matplotlib.ticker import MultipleLocator
from obspy import read, UTCDateTime
from pathlib import Path
from gwpy.timeseries import TimeSeries
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(processName)s] %(message)s")
log = logging.getLogger(__name__)

GPS_UNIX_OFFSET = 315964800
GPS_UTC_OFFSET = 18

df_g = pd.read_csv("gravity_spy_O3b.csv")
df_g["trigger_time"] = df_g["GPStime"].apply(lambda gps: UTCDateTime(gps + GPS_UNIX_OFFSET - GPS_UTC_OFFSET))

PATH = Path("/home/rauls/Desktop/Parallel_test_VRG")
RUNS = ["O3b"]  # ["O3b", "O4b"]
#STATIONS = ["VRG01", "VRG02", "VRG03"]
#CHANNELS = ["HH1", "HH2", "HH3"]

STA = 0.5
LTA = 60
ON_THRESHOLD = 20
OFF_THRESHOLD = 1.5

HALF_WIDTH = 12
FRANGE = (3, 30)
QRANGE = (10, 32)
WHITEN = True

FREQ_BINS = 30
TIME_BINS = 41
INTENSITY_THRESHOLD = 2 * ON_THRESHOLD

OUTPUT_DIR = Path("output")


def generate_qtransform(tr, trigger_time, half_width):
    start = trigger_time - half_width
    end = trigger_time + half_width
    segment = tr.slice(start, end)
    ts = TimeSeries(segment.data, t0=0, sample_rate=segment.stats.sampling_rate)
    qspec = ts.q_transform(qrange=QRANGE, frange=FRANGE, whiten=WHITEN)
    qspec.xindex = qspec.xindex.value - half_width
    return qspec

def qtransform_to_matrix(qspec, interval, nt=TIME_BINS, nf=FREQ_BINS, frange=FRANGE, intensity_threshold=INTENSITY_THRESHOLD):
    power = np.asarray(qspec.value, dtype=float)
    original_times = np.asarray(qspec.xindex.value, dtype=float)
    original_freqs = np.asarray(qspec.yindex.value, dtype=float)

    if power.shape == (len(original_times), len(original_freqs)):
        power = power.T
    elif power.shape == (len(original_freqs), len(original_times)):
        pass
    else:
        raise ValueError(
            "Dimensões incompatíveis entre qspec.value e os eixos da Q-transform:\n"
            f"power.shape = {power.shape}\nlen(time) = {len(original_times)}\nlen(freq) = {len(original_freqs)}")

    freq_mask = np.logical_and(original_freqs >= frange[0], original_freqs <= frange[1])
    original_freqs = original_freqs[freq_mask]
    power = power[freq_mask, :]

    time_edges = np.linspace(-interval, interval, nt + 1)
    freq_edges = np.logspace(np.log10(frange[0]), np.log10(frange[1]), nf + 1)
    matrix = np.zeros((nf, nt), dtype=float)

    for fi in range(nf):
        freq_mask_bin = np.logical_and(original_freqs >= freq_edges[fi], original_freqs < freq_edges[fi + 1])
        if not np.any(freq_mask_bin):
            continue
        for ti in range(nt):
            time_mask_bin = np.logical_and(original_times >= time_edges[ti], original_times < time_edges[ti + 1])
            if not np.any(time_mask_bin):
                continue
            values = power[np.ix_(freq_mask_bin, time_mask_bin)]
            finite_values = values[np.isfinite(values)]
            if finite_values.size > 0:
                matrix[fi, ti] = np.max(finite_values)

    if intensity_threshold is not None:
        matrix[matrix > intensity_threshold] = INTENSITY_THRESHOLD

    return matrix


def plot_qtransform_matrix(matrix, trigger_time, center_time, half_width, output_file, intensity_threshold=INTENSITY_THRESHOLD):
    time_edges = np.linspace(-half_width, half_width, matrix.shape[1] + 1)
    freq_edges = np.logspace(np.log10(FRANGE[0]), np.log10(FRANGE[1]), matrix.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(time_edges, freq_edges, matrix, shading="auto", cmap="jet")
    ax.set_yscale("log")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(freq_edges[0], freq_edges[-1])
    ax.set_xlabel("Time relative to trigger [s]", fontsize=20)
    ax.set_ylabel("Frequency [Hz]", fontsize=20)
    ax.set_title(f"Q-transform Window = ±{HALF_WIDTH} s\n Trigger = {trigger_time}\n Center = {center_time}")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(False)
    mesh.set_edgecolors("face")
    mesh.set_antialiased(False)
    mesh.set_rasterized(True)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Q-transform intensity")
    if intensity_threshold is not None:
        cbar.ax.axhline(intensity_threshold, linestyle="--", linewidth=1.5)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

def parse_date_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(filename))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d")

def get_leaf_folders(data_root):
    """
    Any directory that directly contains .mseed files is treated as a leaf
    task. This avoids hardcoding a fixed depth (run/station/channel/month) -
    it finds your real leaf folders regardless of exactly how deep they sit
    under data_root.
    """
    data_root = Path(data_root)
    return sorted(p for p in data_root.rglob("*") if p.is_dir() and any(p.glob("*.mseed")))

def process_single_file(file_path, run, station, channel, plot_dir):
    """
    Process one mseed file against ONLY the Gravity Spy triggers that fall
    within that file's own time span (previously every file was checked
    against the entire trigger catalog - that's fixed here).
    Returns (rows, stats, failed_file_or_None).
    """
    rows = []
    stats = {"triggers_seen": 0, "triggers_ok": 0, "triggers_failed": 0}

    try:
        st = read(str(file_path), format="mseed")
        tr = st[0]
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime
    except Exception as e:
        log.warning("Failed to read %s: %s", file_path, e)
        return rows, stats, str(file_path)

    file_date = parse_date_from_filename(file_path.name)
    if file_date is None:
        log.warning("Could not parse a date from filename, skipping: %s", file_path.name)
        return rows, stats, str(file_path)

    mask = (df_g["trigger_time"] >= starttime) & (df_g["trigger_time"] <= endtime)
    triggers = df_g.loc[mask, "trigger_time"]
    stats["triggers_seen"] = len(triggers)

    if len(triggers) == 0:
        return rows, stats, None

    day_str = file_date.strftime("%Y%m%d")

    for i, trigger_time in enumerate(triggers):
        try:
            qspec = generate_qtransform(tr, trigger_time, HALF_WIDTH)
            matrix = qtransform_to_matrix(qspec, interval=HALF_WIDTH, nt=TIME_BINS, nf=FREQ_BINS,frange=FRANGE, intensity_threshold=INTENSITY_THRESHOLD)
            
            expected_shape = (FREQ_BINS, TIME_BINS)
            if matrix.shape != expected_shape:
                raise ValueError(f"Unexpected matrix shape: {matrix.shape}. Expected {expected_shape}.")

            qplot_file = plot_dir / f"qtransform_{day_str}_{i:06d}.pdf"
            plot_qtransform_matrix(
                matrix=matrix, trigger_time=trigger_time, center_time=trigger_time,
                half_width=1, output_file=qplot_file, intensity_threshold=INTENSITY_THRESHOLD)

            row = {
                "run": run, "station": station, "channel": channel,
                "file": file_path.name, "date": file_date.strftime("%Y-%m-%d"),
                "trigger_time": str(trigger_time)}
            row.update({f"feature_{j:04d}": v for j, v in enumerate(matrix.flatten())})
            rows.append(row)
            stats["triggers_ok"] += 1
        except Exception as e:
            stats["triggers_failed"] += 1
            log.warning("Trigger %d (%s) failed in %s: %s", i, trigger_time, file_path.name, e)
    return rows, stats, None


def process_leaf_folder(folder_str, data_root_str):
    """
    Runs inside a worker process. Handles every .mseed file in one leaf
    folder, writes that folder's features to its own parquet file, and
    returns summary stats - not raw data. A module-level list like the
    original `rows = []` never makes it back to the parent process, since
    each worker is a separate process, not a thread.
    """
    folder = Path(folder_str)
    data_root = Path(data_root_str)
    rel = folder.relative_to(data_root)
    parts = rel.parts
    run = parts[-4] if len(parts) >= 4 else (RUNS[0] if RUNS else None)
    station = parts[-3] if len(parts) >= 3 else None
    channel = parts[-2] if len(parts) >= 2 else None

    out_file = OUTPUT_DIR / "features" / f"{'__'.join(rel.parts)}.parquet"
    if out_file.exists():  # resumability: skip work already done
        return {"folder": folder_str, "status": "skipped"}

    plot_dir = OUTPUT_DIR / "plots" / rel
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    folder_stats = {"files": 0, "failed_files": [], "triggers_seen": 0, "triggers_ok": 0, "triggers_failed": 0}
    status, error = "ok", None

    try:
        for file_path in sorted(folder.glob("*.mseed")):
            folder_stats["files"] += 1
            rows, stats, failed_file = process_single_file(file_path, run, station, channel, plot_dir)
            all_rows.extend(rows)
            for k in ("triggers_seen", "triggers_ok", "triggers_failed"):
                folder_stats[k] += stats[k]
            if failed_file:
                folder_stats["failed_files"].append(failed_file)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        if all_rows:
            pd.DataFrame(all_rows).to_parquet(out_file)
    except Exception as e:
        status, error = "error", f"{e}\n{traceback.format_exc()}"
    finally:
        gc.collect()

    return {"folder": folder_str, "status": status, "error": error, **folder_stats}


def main():
    data_root = PATH  # adjust if your real leaf folders sit under PATH / RUNS[0] / ...
    leaves = get_leaf_folders(data_root)
    log.info("found %d leaf folders under %s", len(leaves), data_root)
    max_workers = 4

    completed = failed = skipped = 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=5) as executor:
        futures = {executor.submit(process_leaf_folder, str(f), str(data_root)): f for f in leaves}

        for future in as_completed(futures):
            folder = futures[future]
            try:
                result = future.result()
            except Exception:
                failed += 1
                log.exception("Worker process crashed on %s", folder)
                continue

            if result["status"] == "ok":
                completed += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
                log.error("Failed %s: %s", folder, result.get("error"))

            done = completed + failed + skipped
            log.info(
                "%d/%d (ok=%d failed=%d skipped=%d) - %.1fm elapsed | last: files=%s ok_triggers=%s",
                done, len(leaves), completed, failed, skipped,
                (time.time() - start) / 60,
                result.get("files"), result.get("triggers_ok"))


if __name__ == "__main__":
    main()