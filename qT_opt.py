import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from pathlib import Path
from gwpy.timeseries import TimeSeries
from pathlib import Path

# Better version of qtest

#######################################################################
# Inputs
#######################################################################

#2022 (FEITO)
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20220101000000_20220131235959.mseed"
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20220201000000_20220228235959.mseed"
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20220301000000_20220331235959.mseed"
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20220401000000_20220430235959.mseed"
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20220501000000_20220531235959.mseed"
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20220601000000_20220630235959.mseed"
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20220701000000_20220731235959.mseed"
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20220801000000_20220831235959.mseed"
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20220901000000_20220930235959.mseed" 
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20221001000000_20221031235959.mseed" 
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20221101000000_20221130235959.mseed" 
#MSEED_FILE = "SENA-files/2022/eida_response_MN-SENA_20221201000000_20221231235959.mseed" 

#2023 (FEITO)
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20230101000000_20230131235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20230201000000_20230228235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20230301000000_20230331235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20230401000000_20230430235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20230501000000_20230531235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20230601000000_20230630235959.mseed"
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20230701000000_20230731235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20230801000000_20230831235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20230901000000_20230930235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20231001000000_20231031235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20231101000000_20231130235959.mseed" 
#MSEED_FILE = "SENA-files/2023/eida_response_MN-SENA_20231201000000_20231231235959.mseed" 

#2024
MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20240101000000_20240131235959.mseed" #continuar daqui, ainda não rodei
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20240201000000_20240229235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20240301000000_20240331235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20240401000000_20240430235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20240501000000_20240531235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20240601000000_20240630235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20240701000000_20240731235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20240801000000_20240831235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20240901000000_20240930235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20241001000000_20241031235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20241101000000_20241130235959.mseed"
# MSEED_FILE = "SENA-files/2024/eida_response_MN-SENA_20241201000000_20241231235959.mseed"


#2025 (FEITO)
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250101000000_20250131235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250201000000_20250228235959.mseed" 
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250301000000_20250331235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250401000000_20250430235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250501000000_20250531235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250601000000_20250630235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250701000000_20250731235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250801000000_20250831235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250901000000_20250930235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20251001000000_20251031235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20251101000000_20251130235959.mseed"
#MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20251201000000_20251231235959.mseed"

STA = 0.5
LTA = 60
ON_THRESHOLD = 20       #20 was used before
OFF_THRESHOLD = 1.5     #1.5 was used before

HALF_WIDTH = 12 # \pm 12s
FRANGE = (3,30)
QRANGE = (10,32) #Optimal value (tested) was this interval
WHITEN = True 

FREQ_BINS = 30
TIME_BINS = 41
INTENSITY_THRESHOLD = None

PERPLEXITY = 30     # For future tsne analysis
DIMENSIONALITY = 2  # For future tsne analysis

#######################################################################
# Getting triggers with obspy 
#######################################################################

# Obspy processing can be done with functional programing

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
    

#triggers = seismic_trig(MSEED_FILE, FRANGE[0], FRANGE[1], UTC=True, p=True)
#print(triggers)

#if len(triggers) == 0:
#    print("No triggers found.")
#    raise SystemExit

#######################################################################
# Q-transform Functions
#######################################################################

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

    #print(f"Q-transform original: {power.shape}")
    #print(f"Time axis: {len(original_times)}")
    #print(f"Frequency axis: {len(original_freqs)}")

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

    #print(f"Q-transform após correção: {power.shape}")

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
        matrix[matrix < intensity_threshold] = 0.0

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



#######################################################################
# Debug 
#######################################################################

if __name__ == "__main__":

    base_dir = Path("results")
    output_dir = base_dir
    counter = 1

    while output_dir.exists():
        output_dir = Path(f"{base_dir}_{counter}")
        counter += 1
    output_dir.mkdir()
    print(f"Pasta criada: {output_dir}")
    summary_file = output_dir / "summary.txt"

    st = read(MSEED_FILE, format="mseed")
    tr = st[0]
    starttime = tr.stats.starttime
    endtime = tr.stats.endtime

    #print(f"File:{MSEED_FILE}")
    #print(f"Start time: {starttime} | End time: {endtime}")
    tr.plot(outfile=f"{output_dir}/mseed_amplitude.pdf")

    def write_both(text, file):
        print(text)
        file.write(text + "\n")

    with open(summary_file, "w") as f:
        write_both(f"Start time: {starttime} \nEnd time: {endtime}", f)
        write_both(f"Inputs: FRANGE = {FRANGE}, QRANGE = {QRANGE}, WHITHEN = {WHITEN}", f)
        write_both(f"Parameters: sta = {STA}, lta = {LTA}, threshold = {ON_THRESHOLD}", f)
        write_both(f"Images: frequency_bins = {FREQ_BINS}, time_bins = {TIME_BINS}, intensity threshold = {INTENSITY_THRESHOLD}", f)

    
    triggers = seismic_trig(MSEED_FILE,FRANGE[0],FRANGE[1],UTC=True,p=True)

    if len(triggers) == 0:
        print("No triggers found.")
        raise SystemExit

    rows = []
    successful_triggers = 0
    failed_triggers = 0

    for i, trigger_time in enumerate(triggers):

        print(f"\nProcessing trigger {i + 1}/{len(triggers)}: {trigger_time}")

        try:
            qspec = generate_qtransform(tr,trigger_time,HALF_WIDTH)
            matrix = qtransform_to_matrix(qspec,interval=HALF_WIDTH,nt=TIME_BINS,nf=FREQ_BINS,frange=FRANGE,intensity_threshold=INTENSITY_THRESHOLD)

            expected_shape = (FREQ_BINS,TIME_BINS)
            if matrix.shape != expected_shape:
                raise ValueError(f"Unexpected matrix shape: {matrix.shape}. Expected {expected_shape}.")

            output_file = (output_dir /f"qtransform_{i:06d}.pdf")

            plot_qtransform_matrix(matrix=matrix,trigger_time=trigger_time,center_time=trigger_time,half_width=1,output_file=output_file,intensity_threshold=INTENSITY_THRESHOLD)
            features = matrix.flatten()
            row = {"trigger_time": str(trigger_time)}

            for j, value in enumerate(features):
                row[f"feature_{j:04d}"] = value

            rows.append(row)
            successful_triggers += 1
            print(f"Matrix: {matrix.shape} | Features: {features.shape}")

        except Exception as e:
            failed_triggers += 1
            print(f"Failed trigger {i + 1} ({trigger_time}): {e}")

    # ------------------------------------------------------------------------
    # Create DataFrame
    # ------------------------------------------------------------------------

    if len(rows) == 0:
        print("No Q-transforms were successfully processed.")
        raise SystemExit

    df = pd.DataFrame(rows)

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print(f"Total triggers: {len(triggers)}")
    print(f"Failed triggers: {failed_triggers}")
    print(f"DataFrame shape: {df.shape}")

    # ------------------------------------------------------------------------
    # Save Parquet
    # ------------------------------------------------------------------------

    parquet_file = (output_dir / "qtransform_features.parquet")
    df.to_parquet(parquet_file,index=False)

    print()
    print(f"Parquet saved to: {parquet_file}")

    # ------------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------------

    summary_file = (output_dir / "summary.txt")

    with open(summary_file,"w") as f:
        f.write(f"File: {MSEED_FILE}\n")
        f.write(f"Start time: {starttime} | End time: {endtime}\n")
        f.write(f"FRANGE: {FRANGE} | QRANGE: {QRANGE} | WHITEN: {WHITEN}\n")
        f.write(f"STA: {STA} | LTA: {LTA} | ON_THRESHOLD: {ON_THRESHOLD} | OFF_THRESHOLD: {OFF_THRESHOLD}\n")
        f.write(f"Frequency bins: {FREQ_BINS} | Time bins: {TIME_BINS}\n")
        f.write(f"Total triggers: {len(triggers)} | Successful triggers: {successful_triggers}\n")
        f.write(f"Failed triggers: {failed_triggers}\n")
        #f.write(f"Parquet file: {parquet_file}\n")

    print(f"Summary saved to: {summary_file}")