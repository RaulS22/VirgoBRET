import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from pathlib import Path
from gwpy.timeseries import TimeSeries
from pathlib import Path

#TODO: Check the sena files
#TODO: Check if the parameters are great
#32, 16 testar depois
#Tentar um histograma

#TODO: Test other q values and plot an histogram
#TODO: Use the flatten to make a vector of form [x1,x2,...xn,y1,y2,yn,...]

#Note: This code will be the base for the loop and the vectorization.

# ==========================================================
# USER INPUTS
# ==========================================================

#MSEED_FILE = "eida_response_20250101000000_20250201000000.mseed"
#MSEED_FILE = "14-08-25-Fabi.mseed"
#MSEED_FILE = "22-02-25-Raul.mseed"

MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250201000000_20250228235959.mseed"

#WINDOWS = [1, 2, 5, 10, 20, 30, 40]      # seconds on each side
WINDOWS = [20]
FRANGE = (3, 30)
QRANGE = (4, 10)

PEAK_SEARCH_WINDOW = 0.5    # seconds

WHITEN = True
CENTER_ON_PEAK = True
PLOT_RESULTS = True

base_dir = Path("qTransform")
output_dir = base_dir
counter = 1

while output_dir.exists():
    output_dir = Path(f"{base_dir}_{counter}")
    counter += 1
output_dir.mkdir()
print(f"Pasta criada: {output_dir}")

# ==========================================================
# READ DATA
# ==========================================================
st = read(MSEED_FILE, format="mseed")
tr = st[0]
starttime = tr.stats.starttime
endtime = tr.stats.endtime

print(f"Start time: {starttime} \nEnd time: {endtime}")
tr.plot(outfile=f"{output_dir}/mseed_amplitude.pdf")


# # ==========================================================
# # DOWNLOAD RESPONSE
# # ==========================================================
# client = Client("INGV")
# inv = client.get_stations(
#     network="MN",
#     station="SENA",
#     starttime=starttime,
#     endtime=endtime,
#     level="response"
# )
# 
# # ==========================================================
# # PREPROCESS
# # ==========================================================
# 
# tr.detrend("demean")
# tr.detrend("linear")
# tr.remove_response(inventory=inv,output="VEL",water_level=20)

# ==========================================================
# STA/LTA
# ==========================================================

sta = 0.5
lta = 60

df = tr.stats.sampling_rate
tr_original = tr.copy()

fmin = FRANGE[0]
fmax = FRANGE[1]

tr_band = tr_original.copy()
tr_band.filter("bandpass", freqmin=fmin, freqmax=fmax)
df = tr.stats.sampling_rate

cft = recursive_sta_lta(tr_band.data, int(sta * df), int(lta * df))
on_threshold = 20.0
off_threshold = 1.5

triggers = trigger_onset(cft, on_threshold, off_threshold)
print(f"\nNumber of triggers: {len(triggers)}")

# print(tr)
# print("\nTrace information:")
# print(f"Start time: {tr.stats.starttime}")
# print(f"End time:   {tr.stats.endtime}")
# print(f"Duration:   {(tr.stats.endtime - tr.stats.starttime)/86400:.2f} days")
# print(f"Number of samples: {tr.stats.npts}")
# print(f"Sampling rate: {tr.stats.sampling_rate} Hz")
# 
# print(tr_original)
# print("\nTrace information:")
# print(f"Start time: {tr_original.stats.starttime}")
# print(f"End time:   {tr_original.stats.endtime}")
# print(f"Duration:   {(tr_original.stats.endtime - tr_original.stats.starttime)/86400:.2f} days")
# print(f"Number of samples: {tr_original.stats.npts}")
# print(f"Sampling rate: {tr_original.stats.sampling_rate} Hz")

# ==========================================================
# CONVERT TRIGGERS TO UTCDateTime
# ==========================================================

trigger_times = []

for onset, offset in triggers:
    trigger_time = (tr.stats.starttime +onset / tr.stats.sampling_rate)
    trigger_times.append(trigger_time)
    #print(f"Trigger: {trigger_time} (sample {onset})")

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
        search_start = trigger_time - PEAK_SEARCH_WINDOW
        search_end = trigger_time + PEAK_SEARCH_WINDOW

        if (search_start >= tr.stats.starttime and search_end <= tr.stats.endtime):
            search_trace = tr.slice(starttime=search_start, endtime=search_end)
            if len(search_trace.data) > 0:
                imax = np.argmax(np.abs(search_trace.data))
                center_time = (search_trace.stats.starttime + imax / search_trace.stats.sampling_rate)

    # ======================================================
    # LOOP OVER WINDOW SIZES
    # ======================================================

    for half_width in WINDOWS:
        total_duration = 2 * half_width
        min_freq = FRANGE[0]
        if total_duration < 4.0 / min_freq:
            continue

        # --------------------------------------------------
        # Build perfectly symmetric window
        # --------------------------------------------------

        df = tr.stats.sampling_rate
        center_idx = int((center_time - tr.stats.starttime)* df)
        nwin = int(half_width * df)
        i0 = center_idx - nwin
        i1 = center_idx + nwin

        if i0 < 0 or i1 >= tr.stats.npts:
            continue

        data_event = tr.data[i0:i1]

        if len(data_event) == 0:
            continue

        # --------------------------------------------------
        # Relative time:
        # trigger is exactly t = 0
        # --------------------------------------------------

        ts = TimeSeries(data_event.astype(np.float64),sample_rate=df,t0=0)

        # --------------------------------------------------
        # Q-transform
        # --------------------------------------------------

        try:
            qspec = ts.q_transform(frange=FRANGE,qrange=QRANGE,whiten=WHITEN) #frange=FRANGE,qrange=QRANGE,whiten=True
            qspec.xindex = qspec.xindex.value - half_width

        except Exception as e:
            print(f"Q-transform failed:\n{e}")
            continue

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        power = qspec.value
        peak_energy = np.nanmax(power)
        mean_energy = np.nanmean(power)

        results.append({
            "trigger_time": trigger_time,
            "center_time": center_time,
            "half_width": half_width,
            "peak_energy": peak_energy,
            "mean_energy": mean_energy
        })

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        if PLOT_RESULTS:
            fig = qspec.plot()
            ax = fig.axes[0]
            ax.set_title(f"Q-transform Window = ±{half_width} s\n Trigger = {center_time}")
            ax.set_xlabel("Time relative to trigger [s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_yscale("log")
            ax.set_ylim(FRANGE[0], FRANGE[1])
            ticks = [3, 4, 5, 6, 8, 10, 20, 30] # Explicit log ticks
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t) for t in ticks])
            ax.set_xlim(-1.0,1.0)
            ax.axvline(0,color="red",linestyle="--",linewidth=1.5,alpha=0.8)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.grid(False)
            mesh = ax.collections[0]
            #mesh.set_clim(vmin=0, vmax=2*on_threshold)
            mesh.set_edgecolors('face')
            mesh.set_antialiased(False)
            mesh.set_rasterized(True)
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label("Q-transform intensity")
            filename = (f"trigger_{i:04d}_window_{half_width}s.pdf")
            fig.savefig(output_dir / filename, dpi=300) #bbox_inches="tight"
            plt.close(fig)

# ==========================================================
# SUMMARY
# ==========================================================

summary_file = output_dir / "summary.txt"

def write_both(text, file):
    print(text)
    file.write(text + "\n")

with open(summary_file, "w") as f:
    write_both(f"Start time: {starttime} \nEnd time: {endtime}", f)
    write_both(f"Inputs: WINDOWS ={WINDOWS}, FRANGE = {FRANGE}, QRANGE = {QRANGE}, PEAK_SEARCH_WINDOW = {PEAK_SEARCH_WINDOW}, WHITHEN = {WHITEN}", f)
    write_both(f"Parameters: sta = {sta}, lta = {lta}, on_threshold = {on_threshold}, off_threshold = {off_threshold}", f)
    write_both("=" * 60, f)
    write_both("SUMMARY", f)
    write_both("=" * 60, f)

    for r in results:
        write_both(f"Trigger={r['trigger_time']} | Window=±{r['half_width']}s | Peak={r['peak_energy']:.4e} | Mean={r['mean_energy']:.4e}", f)