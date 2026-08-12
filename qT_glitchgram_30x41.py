import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from obspy import read
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from pathlib import Path
from gwpy.timeseries import TimeSeries

#code made by chatgpt 

# ==========================================================
# USER INPUTS
# ==========================================================

#MSEED_FILE = "VirgoBRET/22-02-25-Raul.mseed"
MSEED_FILE ="SENA-files/2025/eida_response_MN-SENA_20250201000000_20250228235959.mseed"

WINDOWS = [12]
FRANGE = (3, 30)
QRANGE = (10, 32)

# STA/LTA parameters
sta = 0.5
lta = 60
on_threshold = 10
off_threshold = 1.5


PEAK_SEARCH_WINDOW = 0.5
WHITEN = True
CENTER_ON_PEAK = True
PLOT_RESULTS = True

# ----------------------------------------------------------
# Fixed image size
# ----------------------------------------------------------

NT = 41       # time bins
NF = 30       # frequency bins

# ----------------------------------------------------------
# Intensity threshold
# ----------------------------------------------------------
# Set to None to keep all intensities.
# Example: 10.0 keeps only pixels with intensity >= 10.
INTENSITY_THRESHOLD = None

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

base_dir = Path("qTransform_31x40")
output_dir = base_dir
counter = 1

while output_dir.exists():
    output_dir = Path(f"{base_dir}_{counter}")
    counter += 1

output_dir.mkdir(parents=True)
print(f"Pasta criada: {output_dir}")


# ==========================================================
# FUNCTION: Q-TRANSFORM -> FIXED 30 x 41 MATRIX
# ==========================================================

def qtransform_to_matrix(qspec, interval, nt=41, nf=30, frange=(3, 30), intensity_threshold=None):
    """
    Converte uma Q-transform do GWpy para uma matriz fixa
    de tamanho (nf, nt) = (30, 41).

    matrix[frequency_bin, time_bin]

    Se houver vários pixels originais dentro de um mesmo
    bin, é utilizada a intensidade máxima.
    """

    # ======================================================
    # 1. Dados da Q-transform
    # ======================================================

    power = np.asarray(qspec.value, dtype=float)
    original_times = np.asarray(qspec.xindex.value, dtype=float)
    original_freqs = np.asarray(qspec.yindex.value, dtype=float)

    print(f"Q-transform original: {power.shape}")
    print(f"Time axis: {len(original_times)}")
    print(f"Frequency axis: {len(original_freqs)}")

    # ======================================================
    # 2. Corrigir orientação da matriz
    # ======================================================
    # Queremos:
    # power.shape = (frequency, time)
    # Se a Q-transform vier como:
    # (time, frequency)
    # fazemos transpose.
    # ======================================================

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

    print(f"Q-transform após correção: {power.shape}")

    # ======================================================
    # 3. Selecionar faixa de frequência
    # ======================================================

    freq_mask = np.logical_and(original_freqs >= frange[0], original_freqs <= frange[1])
    original_freqs = original_freqs[freq_mask]
    power = power[freq_mask,:]

    # ======================================================
    # 4. Bins temporais
    # ======================================================

    time_edges = np.linspace(-interval, interval, nt + 1)
    time_bins = (time_edges[:-1] + time_edges[1:]) / 2.0

    # ======================================================
    # 5. Bins de frequência
    #
    # Logarítmicos
    # ======================================================

    freq_edges = np.logspace(np.log10(frange[0]), np.log10(frange[1]), nf + 1)

    # Média geométrica para os centros
    freq_bins = np.sqrt(freq_edges[:-1] * freq_edges[1:])

    # ======================================================
    # 6. Criar matriz final
    # ======================================================

    matrix = np.zeros((nf, nt),dtype=float)

    # ======================================================
    # 7. Rebinning
    # ======================================================

    for fi in range(nf):
        freq_mask_bin = np.logical_and(
            original_freqs >= freq_edges[fi],
            original_freqs < freq_edges[fi + 1]
        )

        if not np.any(freq_mask_bin):
            continue

        for ti in range(nt):

            time_mask_bin = np.logical_and(
                original_times >= time_edges[ti],
                original_times < time_edges[ti + 1]
            )

            if not np.any(time_mask_bin):
                continue

            values = power[np.ix_(freq_mask_bin,time_mask_bin)]
            finite_values = values[np.isfinite(values)]
            if finite_values.size > 0:
                matrix[fi, ti] = np.max(finite_values)

    # ======================================================
    # 8. Aplicar threshold
    # ======================================================

    if intensity_threshold is not None:
        matrix[matrix < intensity_threshold] = 0.0

    return (matrix,time_bins,freq_bins,time_edges,freq_edges)


# ==========================================================
# FUNCTION: PLOT FIXED MATRIX
# ==========================================================

def plot_qtransform_matrix(matrix, time_edges, freq_edges, trigger_time, center_time, half_width, output_file,
    intensity_threshold=None):
    """
    Plot and save the fixed-size Q-transform matrix.
    """

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(time_edges, freq_edges, matrix, shading="auto", cmap="jet")
    ax.set_yscale("log")
    ax.set_xlim(-1,1)
    ax.set_ylim(freq_edges[0],freq_edges[-1])
    ax.set_xlabel("Time relative to trigger [s]", fontsize=20)
    ax.set_ylabel("Frequency [Hz]", fontsize=20)
    ax.set_title(f"Q-transform Window = ±{half_width} s\n Trigger = {trigger_time}\n Center = {center_time}")
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


# ==========================================================
# READ DATA
# ==========================================================

st = read(MSEED_FILE,format="mseed")
tr = st[0]
starttime = tr.stats.starttime
endtime = tr.stats.endtime

print(f"Start time: {starttime}\n End time: {endtime}")
tr.plot(outfile=output_dir / "mseed_amplitude.pdf")


# ==========================================================
# STA/LTA
# ==========================================================

df = tr.stats.sampling_rate
tr_original = tr.copy()
tr_band = tr_original.copy()
cft = recursive_sta_lta(tr_band.data,int(sta * df),int(lta * df))
triggers = trigger_onset(cft, on_threshold, off_threshold)
print(f"Number of triggers: {len(triggers)}")

# ==========================================================
# CONVERT TRIGGERS TO UTCDateTime
# ==========================================================

trigger_times = []
for onset, offset in triggers:
    trigger_time = (tr.stats.starttime + onset / tr.stats.sampling_rate)
    trigger_times.append(trigger_time)

trigger_times = sorted(trigger_times)

if len(trigger_times) == 0:
    print("No triggers found.")
    raise SystemExit


# ==========================================================
# Q-TRANSFORM ANALYSIS
# ==========================================================

results = []

for i, trigger_time in enumerate(trigger_times):
    center_time = trigger_time

    # ------------------------------------------------------
    # Optional peak centering
    # ------------------------------------------------------

    if CENTER_ON_PEAK:
        search_start = (trigger_time - PEAK_SEARCH_WINDOW)
        search_end = (trigger_time + PEAK_SEARCH_WINDOW)
        if (search_start >= tr.stats.starttime and search_end <= tr.stats.endtime):
            search_trace = tr.slice(starttime=search_start, endtime=search_end)
            if len(search_trace.data) > 0:
                imax = np.argmax(np.abs(search_trace.data))
                center_time = (search_trace.stats.starttime + imax / search_trace.stats.sampling_rate)

    # ======================================================
    # LOOP OVER WINDOW SIZES
    # ======================================================

    for half_width in WINDOWS:
        total_duration = (2 * half_width)
        min_freq = FRANGE[0]

        if total_duration < 4.0 / min_freq:
            continue

        # --------------------------------------------------
        # Build perfectly symmetric window
        # --------------------------------------------------

        df = tr.stats.sampling_rate
        center_idx = int((center_time - tr.stats.starttime) * df)
        nwin = int(half_width * df)
        i0 = center_idx - nwin
        i1 = center_idx + nwin

        if (i0 < 0 or i1 >= tr.stats.npts):
            continue

        data_event = tr.data[i0:i1]
        if len(data_event) == 0:
            continue

        # --------------------------------------------------
        # Relative time:
        # center of event = t = 0
        # --------------------------------------------------

        ts = TimeSeries(data_event.astype(np.float64), sample_rate=df, t0=0)

        # ==================================================
        # Q-TRANSFORM
        # ==================================================

        try:
            qspec = ts.q_transform(frange=FRANGE, qrange=QRANGE, whiten=WHITEN)
            qspec.xindex = (qspec.xindex.value - half_width)

        except Exception as e:
            print(f"Q-transform failed:\n{e}")
            continue

        # ==================================================
        # ORIGINAL Q-TRANSFORM METRICS
        # ==================================================

        power = np.asarray(qspec.value, dtype=float)
        peak_energy = np.nanmax(power)
        mean_energy = np.nanmean(power)

        # ==================================================
        # CONVERT TO 30 x 41 MATRIX
        # ==================================================

        (matrix,time_bins,freq_bins, time_edges,freq_edges) = qtransform_to_matrix( qspec=qspec, interval=1.0, nt=NT, nf=NF, frange=FRANGE, intensity_threshold=(INTENSITY_THRESHOLD))

        # --------------------------------------------------
        # Store result
        # --------------------------------------------------

        results.append({
            "trigger_time": trigger_time,
            "center_time": center_time,
            "half_width": half_width,
            "peak_energy": peak_energy,
            "mean_energy": mean_energy,
            "matrix": matrix})

        # ==================================================
        # SAVE MATRIX
        # ==================================================

        matrix_filename = (f"trigger_{i:04d}_window_{half_width}s_matrix.npy")
        np.save(output_dir / matrix_filename,matrix)

        # ==================================================
        # SAVE MATRIX AS CSV
        # ==================================================

        csv_filename = (f"trigger_{i:04d}_window_{half_width}s_matrix.csv")
        np.savetxt(output_dir / csv_filename, matrix, delimiter=",")

        # ==================================================
        # PLOT ORIGINAL Q-TRANSFORM
        # ==================================================

        if PLOT_RESULTS:
            fig = qspec.plot()
            ax = fig.axes[0]
            ax.set_title(f"Q-transform Window = ±{half_width} s\n Trigger = {center_time}")
            ax.set_xlabel("Time relative to trigger [s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_yscale("log")
            ax.set_ylim(FRANGE[0],FRANGE[1])
            ax.set_xlim(-1,1)
            ax.axvline(0,color="red",linestyle="--",linewidth=1.5,alpha=0.8)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.grid(False)
            mesh = ax.collections[0]
            mesh.set_edgecolors("face")
            mesh.set_antialiased(False)
            mesh.set_rasterized(True)
            cbar = fig.colorbar(mesh,ax=ax)
            cbar.set_label("Q-transform intensity")
            filename = (f"trigger_{i:04d}_window_{half_width}s.pdf")
            fig.savefig(output_dir / filename,dpi=300)
            plt.close(fig)

        # ==================================================
        # PLOT FIXED 30 x 41 MATRIX
        # ==================================================

        matrix_filename_pdf = (f"trigger_{i:04d}_window_{half_width}s_30x41.pdf")
        plot_qtransform_matrix(matrix=matrix,time_edges=time_edges,freq_edges=freq_edges,trigger_time=trigger_time,
                               center_time=center_time,half_width=half_width,output_file=(output_dir/ matrix_filename_pdf),
            intensity_threshold=(INTENSITY_THRESHOLD))


# ==========================================================
# SUMMARY
# ==========================================================

summary_file = (output_dir / "summary.txt")
def write_both(text, file):
    print(text)
    file.write(text + "\n")

with open(summary_file, "w") as f:
    write_both(f"Start time: {starttime}\n End time: {endtime}", f)
    write_both("Inputs: WINDOWS={WINDOWS}, FRANGE={FRANGE}, QRANGE={QRANGE}, PEAK_SEARCH_WINDOW={PEAK_SEARCH_WINDOW}, WHITEN={WHITEN}",f)
    write_both(f"Parameters: sta={sta}, lta={lta}, on_threshold={on_threshold}, off_threshold={off_threshold}",f)
    write_both(f"Matrix size: {NF} frequency bins x {NT} time bins",f)
    write_both(f"Intensity threshold: {INTENSITY_THRESHOLD}", f)
    write_both("=" * 60,f)
    write_both("SUMMARY",f)
    write_both("=" * 60,f)
    for r in results:
        write_both(f"Trigger={r['trigger_time']} | Center={r['center_time']} | Window=±{r['half_width']}s | Peak={r['peak_energy']:.4e} | Mean={r['mean_energy']:.4e} | Matrix={r['matrix'].shape}", f)

print("\nProcessing finished.")
print(f"Number of processed Q-transforms: {len(results)}")
print(f"Output directory: {output_dir}")
