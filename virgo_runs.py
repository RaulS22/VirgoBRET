import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from datetime import datetime
from matplotlib.ticker import MultipleLocator
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from pathlib import Path
from gwpy.timeseries import TimeSeries

"""
Before we proceed, it is important to point out that the structure of the folders is the following:
---Virgo_data
    |---O3b
        |---VRG01_HH1_[date].mseed
        |---VRG02_HH1_[date].mseed
        |---VRG03_HH1_[date].mseed
        |---VRG01_HH2_[date].mseed
        |---VRG02_HH2_[date].mseed
        |---VRG03_HH2_[date].mseed
        .
        .
        .
        |---VRG01_HH3_[date].mseed
        |---VRG02_HH3_[date].mseed
        |---VRG03_HH3_[date].mseed
    |---O4b
        |---VRG01_HH1_[date].mseed
        .
        .
        .
        |---VRG03_HH3_[date].mseed

        
and this is a direct result of the script download_virgo.sh, which comphends data of all days for the O3b
"2019-11-01T15:00:00Z"~"2020-03-27T17:00:00Z" and O4b "2024-04-10T15:00:00Z"~"2025-01-28T17:00:00Z" runs.
"""

# Global parameters (the same used for SoS-Enatos mine)
PATH = Path("VirgoBRET/Virgo_data")

YEARS    = ["2022", "2023", "2024", "2025"] #["O3b", "O4b"]
STATIONS = ["VRG01", "VRG02", "VRG03"]
CHANNELS = ["HH1", "HH2", "HH3"]

STA = 0.5
LTA = 60
ON_THRESHOLD = 20      
OFF_THRESHOLD = 1.5   

HALF_WIDTH = 12 
FRANGE = (3,30)
QRANGE = (10,32) 
WHITEN = True 

FREQ_BINS = 30
TIME_BINS = 41
INTENSITY_THRESHOLD = 2*ON_THRESHOLD

# Same functions defined at the code qT_opt.py

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
        matrix[matrix > intensity_threshold] = INTENSITY_THRESHOLD

    return matrix

def plot_qtransform_matrix(matrix, trigger_time, center_time, half_width, output_file, intensity_threshold=INTENSITY_THRESHOLD):
    time_edges = np.linspace(-half_width, half_width, matrix.shape[1] + 1)
    freq_edges = np.logspace(np.log10(FRANGE[0]), np.log10(FRANGE[1]), matrix.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(time_edges, freq_edges, matrix, shading="auto", cmap="jet")
    ax.set_yscale("log")
    ax.set_xlim(-1.0,1.0)
    ax.set_ylim(freq_edges[0],freq_edges[-1])
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
        cbar.ax.axhline(intensity_threshold,linestyle="--",linewidth=1.5)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
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

#######################################################################
# Main: loop over every run / station / channel / day-file
#######################################################################

#TODO: use concurrent.futures.ProcessPoolExecutor for peformance

if __name__ == "__main__":
    base_dir = Path("year_Virgo_results_cut")
    output_dir = base_dir
    counter = 1

    while output_dir.exists():
        output_dir = Path(f"{base_dir}_{counter}")
        counter += 1
    output_dir.mkdir()
    print(f"Pasta criada: {output_dir}")

    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Inputs: FRANGE = {FRANGE}, QRANGE = {QRANGE}, WHITEN = {WHITEN}\n")
        f.write(f"Parameters: sta = {STA}, lta = {LTA}, on_threshold = {ON_THRESHOLD}, off_threshold = {OFF_THRESHOLD}\n")
        f.write(f"Images: frequency_bins = {FREQ_BINS}, time_bins = {TIME_BINS}, intensity_threshold = {INTENSITY_THRESHOLD}\n")

    rows = []
    total_files = 0
    failed_files = []
    total_triggers = 0
    successful_triggers = 0
    failed_triggers = 0

    for year in YEARS:
        run_dir = PATH / year

        if not run_dir.exists():
            print(f"Run folder not found, skipping: {run_dir}")
            continue

        for station in STATIONS:
            for channel in CHANNELS:

                pattern = f"{station}_{channel}_*.mseed"
                files = sorted(run_dir.glob(pattern))

                if not files:
                    print(f"No files found for {year}/{station}_{channel}")
                    continue

                for mseed_file in files:
                    total_files += 1
                    #print(f"\n{'=' * 70}")
                    #print(f"Processing: {mseed_file}")
                    #print(f"{'=' * 70}")

                    try:
                        st = read(mseed_file, format="mseed")
                        tr = st[0]
                        starttime = tr.stats.starttime
                        endtime = tr.stats.endtime
                    except Exception as e:
                        print(f"Failed to read {mseed_file}: {e}")
                        failed_files.append(str(mseed_file))
                        continue

                    file_date = parse_date_from_filename(mseed_file.name)
                    if file_date is None:
                        print(f"Could not parse a date from filename, skipping: {mseed_file.name}")
                        failed_files.append(str(mseed_file))
                        continue

                    triggers = seismic_trig(mseed_file, FRANGE[0], FRANGE[1], UTC=True, p=True)

                    if len(triggers) == 0:
                        print("No triggers found.")
                        continue

                    total_triggers += len(triggers)
                    month_year = file_date.strftime("%b_%Y").lower()
                    day_str = file_date.strftime("%Y%m%d")
                    file_out_dir = output_dir / year / station / channel / month_year
                    file_out_dir.mkdir(parents=True, exist_ok=True)

                    for i, trigger_time in enumerate(triggers):
                        #print(f"Processing trigger {i + 1}/{len(triggers)}: {trigger_time}")
                        try:
                            qspec = generate_qtransform(tr, trigger_time, HALF_WIDTH)
                            matrix = qtransform_to_matrix(qspec,interval=HALF_WIDTH,nt=TIME_BINS,nf=FREQ_BINS,frange=FRANGE,intensity_threshold=INTENSITY_THRESHOLD)

                            expected_shape = (FREQ_BINS, TIME_BINS)
                            if matrix.shape != expected_shape:
                                raise ValueError(f"Unexpected matrix shape: {matrix.shape}. Expected {expected_shape}.")

                            qplot_file = file_out_dir / f"qtransform_{day_str}_{i:06d}.pdf"
                            plot_qtransform_matrix(matrix=matrix,trigger_time=trigger_time,center_time=trigger_time,half_width=1,output_file=qplot_file,intensity_threshold=INTENSITY_THRESHOLD)

                            features = matrix.flatten()
                            row = {"year": year,"station": station,"channel": channel,"file": mseed_file.name,"date": file_date.strftime("%Y-%m-%d"),"month_year": month_year,"trigger_time": str(trigger_time)}
                            for j, value in enumerate(features):
                                row[f"feature_{j:04d}"] = value

                            rows.append(row)
                            successful_triggers += 1
                            print(f"Matrix: {matrix.shape} | Features: {features.shape}")

                        except Exception as e:
                            failed_triggers += 1
                            print(f"Failed trigger {i + 1} ({trigger_time}): {e}")

    # ------------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)
    print(f"Total files processed: {total_files}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Total triggers: {total_triggers}")
    print(f"Successful triggers: {successful_triggers}")
    print(f"Failed triggers: {failed_triggers}")

    if len(rows) == 0:
        print("No Q-transforms were successfully processed.")
        raise SystemExit

    df = pd.DataFrame(rows)
    print(f"DataFrame shape: {df.shape}")

    # ------------------------------------------------------------------------
    # Save Parquet
    # ------------------------------------------------------------------------

    parquet_file = output_dir / "qtransform_features.parquet"
    df.to_parquet(parquet_file, index=False)
    print()
    print(f"Parquet saved to: {parquet_file}")

    # ------------------------------------------------------------------------
    # Update summary
    # ------------------------------------------------------------------------

    with open(summary_file, "a") as f:
        f.write(f"\nTotal files processed: {total_files} | Failed files: {len(failed_files)}\n")
        f.write(f"Total triggers: {total_triggers} | Successful triggers: {successful_triggers} | Failed triggers: {failed_triggers}\n")
        f.write(f"DataFrame shape: {df.shape}\n")
        if failed_files:
            f.write("Failed files:\n")
            for ff in failed_files:
                f.write(f"  {ff}\n")

    print(f"Summary saved to: {summary_file}")