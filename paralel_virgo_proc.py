import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from pathlib import Path
from gwpy.timeseries import TimeSeries


# Global parameters (the same used for SoS-Enatos mine)
PATH = Path("year_Virgo_data")

YEARS    = ["2022", "2023", "2024", "2025"] #["O3b", "O4b"]
STATIONS = ["VRG01"]
CHANNELS = ["HH3"]

STA = 0.5
LTA = 60
ON_THRESHOLD = 20
OFF_THRESHOLD = 1.5

HALF_WIDTH = 12
MATRIX_HALF_WIDTHS = [1.0, 2.0, 4.0, 6.0, 12.0]   
MAX_WORKERS = len(MATRIX_HALF_WIDTHS)
FRANGE = (2,30)
QRANGE = (10,32)
WHITEN = True

NATURAL_CMAP = "viridis"   # colormap for the 3rd page (matplotlib/gwpy default); page 1 and 2 keep "jet"

FREQ_BINS = 30
TIME_BINS = 41
INTENSITY_THRESHOLD = 2*ON_THRESHOLD

PARQUET_CHUNK_SIZE = 200   # write one parquet part every this many successfully processed triggers

def seismic_trig(file, fmin, fmax, UTC=False, p=False):
    st = read(file, format="mseed")
    tr = st[0]
    starttime = tr.stats.starttime
    endtime = tr.stats.endtime
    df = tr.stats.sampling_rate
    tr_original = tr.copy()
    tr_band = tr_original.copy()
    tr_band.filter("bandpass", freqmin=fmin, freqmax=fmax)
    df = tr.stats.sampling_rate
    cft = recursive_sta_lta(tr_band.data, int(STA * df), int(LTA * df))
    triggers = trigger_onset(cft, ON_THRESHOLD, OFF_THRESHOLD)
    if p==True:
            print(f"\nNumber of triggers: {len(triggers)}")

    if UTC==True:
        trigger_times = []
        for onset, offset in triggers:
            trigger_time = (tr.stats.starttime +onset / tr.stats.sampling_rate)
            trigger_times.append(trigger_time)
        triggers = sorted(trigger_times)
    return triggers

def generate_qtransform(tr, trigger_time, half_width):
    start = trigger_time - half_width
    end = trigger_time + half_width
    segment = tr.slice(start, end)

    #t0 = 0  #t0=segment.stats.starttime.timestamp
    ts = TimeSeries(segment.data,t0=0,sample_rate=segment.stats.sampling_rate)
    qspec = ts.q_transform(qrange=QRANGE,frange=FRANGE,whiten=WHITEN)
    qspec.xindex = qspec.xindex.value - half_width

    return qspec

def qtransform_to_matrix(qspec, interval, nt=TIME_BINS, nf=FREQ_BINS, frange=FRANGE, intensity_threshold=INTENSITY_THRESHOLD):
    power = np.asarray(qspec.value, dtype=float)
    original_times = np.asarray(qspec.xindex.value, dtype=float)
    original_freqs = np.asarray(qspec.yindex.value, dtype=float)

    if power.shape == (len(original_times),len(original_freqs)):
        power = power.T

    elif power.shape == (len(original_freqs),len(original_times)):
        pass

    else:
        raise ValueError(
            "Dimensões incompatíveis entre qspec.value "
            "e os eixos da Q-transform:\n"
            f"power.shape = {power.shape}\n"
            f"len(time) = {len(original_times)}\n"
            f"len(freq) = {len(original_freqs)}"
        )

    freq_mask = np.logical_and(original_freqs >= frange[0], original_freqs <= frange[1])
    original_freqs = original_freqs[freq_mask]
    power = power[freq_mask,:]

    time_edges = np.linspace(-interval, interval, nt + 1)
    time_bins = (time_edges[:-1] + time_edges[1:]) / 2.0

    freq_edges = np.logspace(np.log10(frange[0]), np.log10(frange[1]), nf + 1)
    freq_bins = np.sqrt(freq_edges[:-1] * freq_edges[1:])
    matrix = np.zeros((nf, nt),dtype=float)

    for fi in range(nf):
        freq_mask_bin = np.logical_and(
            original_freqs >= freq_edges[fi],
            original_freqs < freq_edges[fi + 1])

        if not np.any(freq_mask_bin):
            continue

        for ti in range(nt):

            time_mask_bin = np.logical_and(
                original_times >= time_edges[ti],
                original_times < time_edges[ti + 1])

            if not np.any(time_mask_bin):
                continue

            values = power[np.ix_(freq_mask_bin,time_mask_bin)]
            finite_values = values[np.isfinite(values)]
            if finite_values.size > 0:
                matrix[fi, ti] = np.max(finite_values)

    if intensity_threshold is not None:
        matrix[matrix > intensity_threshold] = intensity_threshold

    return matrix

def _matrix_figure(matrix, trigger_time, center_time, half_width, intensity_threshold=INTENSITY_THRESHOLD):
    """Page 1: the binned (FREQ_BINS x TIME_BINS) matrix, saturated at intensity_threshold."""
    time_edges = np.linspace(-half_width, half_width, matrix.shape[1] + 1)
    freq_edges = np.logspace(np.log10(FRANGE[0]), np.log10(FRANGE[1]), matrix.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(time_edges, freq_edges, matrix, shading="auto", cmap="jet", vmin=0, vmax=intensity_threshold)
    ax.set_yscale("log")
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(freq_edges[0],freq_edges[-1])
    ax.set_xlabel("Time relative to trigger [s]", fontsize=20)
    ax.set_ylabel("Frequency [Hz]", fontsize=20)
    ax.set_title(f"Binned matrix ({matrix.shape[0]} x {matrix.shape[1]}) | shown ±{half_width} s (Q-transform computed on ±{HALF_WIDTH} s)\n Trigger = {trigger_time}\n Center = {center_time}")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    tick_step = 0.5 if half_width <= 2 else (1.0 if half_width <= 6 else 2.0)   # keeps ~9-13 ticks for any window
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.grid(False)
    mesh.set_edgecolors("face")
    mesh.set_antialiased(False)
    mesh.set_rasterized(True)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Q-transform intensity")

    if intensity_threshold is not None:
        cbar.ax.axhline(intensity_threshold,linestyle="--",linewidth=1.5)

    fig.tight_layout()
    return fig


def _spectrogram_figure(qspec, trigger_time, center_time, xlim=None, intensity_threshold=INTENSITY_THRESHOLD, cmap="jet", tag=""):
    """
    Page 2: the actual Q-transform spectrogram at its native resolution, saturated at
    intensity_threshold exactly like the matrix (same cmap, vmin/vmax and colorbar marker).
    xlim=None shows the full +-HALF_WIDTH window.
    """
    power = np.asarray(qspec.value, dtype=float)
    times = np.asarray(qspec.xindex.value, dtype=float)
    freqs = np.asarray(qspec.yindex.value, dtype=float)

    # Same orientation logic as qtransform_to_matrix: we want (n_freq, n_time)
    if power.shape == (len(times), len(freqs)):
        power = power.T
    elif power.shape != (len(freqs), len(times)):
        raise ValueError(f"Incompatible qspec shape {power.shape} vs time={len(times)}, freq={len(freqs)}")

    freq_mask = np.logical_and(freqs >= FRANGE[0], freqs <= FRANGE[1])
    freqs = freqs[freq_mask]
    power = power[freq_mask, :]

    # Saturate: everything above the threshold is set to the threshold (as in the matrix)
    # Percentage of clipped pixels is computed only over the time window that is displayed
    x0, x1 = xlim if xlim is not None else (times[0], times[-1])
    shown = power[:, np.logical_and(times >= x0, times <= x1)]
    saturated_pct = 100.0 * np.sum(np.isfinite(shown) & (shown > intensity_threshold)) / max(np.isfinite(shown).sum(), 1)
    power_sat = np.where(np.isfinite(power), np.minimum(power, intensity_threshold), 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(times, freqs, power_sat, shading="nearest", cmap=cmap, vmin=0, vmax=intensity_threshold)
    ax.set_yscale("log")
    ax.set_xlim(*(xlim if xlim is not None else (times[0], times[-1])))
    ax.set_ylim(FRANGE[0], FRANGE[1])
    ax.set_xlabel("Time relative to trigger [s]", fontsize=20)
    ax.set_ylabel("Frequency [Hz]", fontsize=20)
    ax.set_title(
        f"Spectrogram (native resolution{tag}) | Q-transform computed on ±{HALF_WIDTH} s\n"
        f" Trigger = {trigger_time}\n Center = {center_time}\n"
        f" Saturated at {intensity_threshold:g} ({saturated_pct:.2f}% of pixels clipped)"
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.grid(False)
    mesh.set_rasterized(True)

    # extend="max" draws the arrow at the top of the colorbar: values above it are clipped
    cbar = fig.colorbar(mesh, ax=ax, extend="max")
    cbar.set_label(f"Q-transform intensity (saturated at {intensity_threshold:g})")
    if intensity_threshold is not None:
        cbar.ax.axhline(intensity_threshold, linestyle="--", linewidth=1.5)

    fig.tight_layout()
    return fig


def plot_qtransform_pdf(qspec, matrix, trigger_time, center_time, half_width, output_file,
                        intensity_threshold=INTENSITY_THRESHOLD, spec_xlim=None):
    """
    Three-page PDF: 1 = binned matrix (jet), 2 = full spectrogram (jet), 3 = full spectrogram (natural cmap). All saturated.
    spec_xlim=None shows the spectrograms over the same +-half_width as the matrix.
    """
    if spec_xlim is None:
        spec_xlim = (-half_width, half_width)

    with PdfPages(output_file) as pdf:
        fig = _matrix_figure(matrix, trigger_time, center_time, half_width, intensity_threshold)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig = _spectrogram_figure(qspec, trigger_time, center_time, spec_xlim, intensity_threshold)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig = _spectrogram_figure(qspec, trigger_time, center_time, spec_xlim, intensity_threshold,
                                  cmap=NATURAL_CMAP, tag=f", {NATURAL_CMAP}")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

# Some adjustments for the output

"""
In order to facilitate the study of periodicity, it is desirable to save the processed data at the
following folders structure:

---processed_Virgo_data
    |---2022
        |---jan_2022
        .
        .
        .
        |---dec_2022
    |---2025
        |---jan_2025
        .
        .
        .
        |---dec_2025

so we can make use of parents_dir name.
"""

def parse_date_from_filename(filename):
    """
    Extracts the YYYY-MM-DD date embedded in a VRGOx_HHx_[date].mseed
    filename and returns it as a datetime. Returns None if no date-like
    substring is found.
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(filename))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d")

def write_parquet_part(rows, parts_dir, part_idx):
    """
    Writes the accumulated rows as one parquet part. The data goes to a hidden temporary file first
    and is renamed when complete, so a killed run never leaves a half-written part behind
    (pyarrow ignores files starting with '.' when the folder is read back as one dataset).
    """
    part_file = parts_dir / f"qtransform_features_part_{part_idx:05d}.parquet"
    tmp_file = parts_dir / f".{part_file.name}.tmp"
    pd.DataFrame(rows).to_parquet(tmp_file, index=False)
    tmp_file.replace(part_file)
    return part_file

#######################################################################
# Output folders: one per window, named from the window value
#######################################################################

OUTPUT_PREFIX = f"{STATIONS[0]}_spectrograms_matrix_results_cut"   # folder name = <OUTPUT_PREFIX>_hw<window>

def window_tag(matrix_half_width):
    """1.0 -> '1', 0.5 -> '0p5', 12.0 -> '12' (used in the folder names and in the log prefix)."""
    return f"{matrix_half_width:g}".replace(".", "p")

def make_output_dir(matrix_half_width):
    """Creates and returns a fresh folder for this window (adds _1, _2, ... if it already exists from an earlier run)."""
    base_dir = Path(f"{OUTPUT_PREFIX}_hw{window_tag(matrix_half_width)}")
    output_dir = base_dir
    counter = 1

    while output_dir.exists():
        output_dir = Path(f"{base_dir}_{counter}")
        counter += 1
    output_dir.mkdir()
    return output_dir

#######################################################################
# Worker: loop over every year / station / channel / day-file for ONE window
#######################################################################

def process_window(matrix_half_width, output_dir):
    """
    Runs the whole pipeline for a single matrix half-width. It runs in its own process and only uses
    its arguments plus the read-only constants at the top, so nothing is shared between windows.
    Returns the counters so the parent process can print a final table.
    """
    tag = window_tag(matrix_half_width)

    def log(message):
        print(f"[hw={tag}] {message}", flush=True)

    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Inputs: FRANGE = {FRANGE}, QRANGE = {QRANGE}, WHITEN = {WHITEN}, HALF_WIDTH = {HALF_WIDTH}, MATRIX_HALF_WIDTH = {matrix_half_width}\n")
        f.write(f"Parameters: sta = {STA}, lta = {LTA}, on_threshold = {ON_THRESHOLD}, off_threshold = {OFF_THRESHOLD}\n")
        f.write(f"Images: frequency_bins = {FREQ_BINS}, time_bins = {TIME_BINS}, intensity_threshold = {INTENSITY_THRESHOLD}\n")

    parts_dir = output_dir / "parquet_parts"
    parts_dir.mkdir()
    part_idx = 0

    rows = []
    total_files = 0
    failed_files = []
    total_triggers = 0
    successful_triggers = 0
    failed_triggers = 0

    for year in YEARS:
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

                    file_date = parse_date_from_filename(mseed_file.name)
                    if file_date is None:
                        log(f"Could not parse a date from filename, skipping: {mseed_file.name}")
                        failed_files.append(str(mseed_file))
                        continue

                    triggers = seismic_trig(mseed_file, FRANGE[0], FRANGE[1], UTC=True, p=False)
                    log(f"Number of triggers: {len(triggers)}")

                    if len(triggers) == 0:
                        log("No triggers found.")
                        continue

                    total_triggers += len(triggers)
                    month_year = file_date.strftime("%b_%Y").lower()
                    day_str = file_date.strftime("%Y%m%d")
                    file_out_dir = output_dir / year / station / channel / month_year
                    file_out_dir.mkdir(parents=True, exist_ok=True)

                    for i, trigger_time in enumerate(triggers):
                        try:
                            qspec = generate_qtransform(tr, trigger_time, HALF_WIDTH)
                            matrix = qtransform_to_matrix(qspec,interval=matrix_half_width,nt=TIME_BINS,nf=FREQ_BINS,frange=FRANGE,intensity_threshold=INTENSITY_THRESHOLD)

                            expected_shape = (FREQ_BINS, TIME_BINS)
                            if matrix.shape != expected_shape:
                                raise ValueError(f"Unexpected matrix shape: {matrix.shape}. Expected {expected_shape}.")

                            qplot_file = file_out_dir / f"qtransform_{day_str}_{i:06d}.pdf"
                            plot_qtransform_pdf(qspec=qspec,matrix=matrix,trigger_time=trigger_time,center_time=trigger_time,half_width=matrix_half_width,output_file=qplot_file,intensity_threshold=INTENSITY_THRESHOLD)

                            features = matrix.flatten()
                            row = {"year": year,"station": station,"channel": channel,"file": mseed_file.name,"date": file_date.strftime("%Y-%m-%d"),"month_year": month_year,"trigger_time": str(trigger_time)}
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

    n_columns = 7 + FREQ_BINS * TIME_BINS   # 7 metadata columns + one column per matrix pixel
    log(f"Total rows: {successful_triggers} in {part_idx} parquet part(s) at {parts_dir}")

    with open(summary_file, "a") as f:
        f.write(f"\nTotal files processed: {total_files} | Failed files: {len(failed_files)}\n")
        f.write(f"Total triggers: {total_triggers} | Successful triggers: {successful_triggers} | Failed triggers: {failed_triggers}\n")
        f.write(f"Parquet: {successful_triggers} rows x {n_columns} columns in {part_idx} part(s) of up to {PARQUET_CHUNK_SIZE} rows, at {parts_dir}\n")
        if failed_files:
            f.write("Failed files:\n")
            for ff in failed_files:
                f.write(f"  {ff}\n")

    log(f"Summary saved to: {summary_file}")
    return result

#######################################################################
# Main: one process per window
#######################################################################

if __name__ == "__main__":
    windows = list(MATRIX_HALF_WIDTHS)

    invalid = [hw for hw in windows if not 0 < hw <= HALF_WIDTH]
    if invalid:
        raise SystemExit(f"Every MATRIX_HALF_WIDTHS value must be in (0, {HALF_WIDTH}] (the Q-transform is only computed on +-{HALF_WIDTH} s). Offending values: {invalid}")
    if len({window_tag(hw) for hw in windows}) != len(windows):
        raise SystemExit(f"MATRIX_HALF_WIDTHS has duplicated values: {windows}")

    # The folders are created here, one after the other, before any process starts (no race between workers)
    output_dirs = {}
    for hw in windows:
        output_dirs[hw] = make_output_dir(hw)
        print(f"Pasta criada: {output_dirs[hw]}")

    results = []
    failed_windows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_window, hw, output_dirs[hw]): hw for hw in windows}

        for future in as_completed(futures):
            hw = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                failed_windows.append(hw)
                print(f"[hw={window_tag(hw)}] WINDOW FAILED with {type(e).__name__}: {e}", flush=True)

    # ------------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)
    for r in sorted(results, key=lambda r: r["window"]):
        print(f"hw = {r['window']:g} s | files {r['total_files']} (failed {r['failed_files']}) | "
              f"triggers {r['total_triggers']} | ok {r['successful_triggers']} | failed {r['failed_triggers']} | "
              f"{r['parquet_parts']} parquet part(s) | {r['output_dir']}")

    if failed_windows:
        print(f"\nWindows that crashed (see the messages above): {sorted(failed_windows)}")
        raise SystemExit(1)