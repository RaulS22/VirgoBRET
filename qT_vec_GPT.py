import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from obspy import read
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from pathlib import Path
from gwpy.timeseries import TimeSeries
from scipy.interpolate import RegularGridInterpolator

# ==========================================================
# USER INPUTS
# ==========================================================

MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250301000000_20250331235959.mseed"

WINDOWS = [20]

FRANGE = (3,30)
QRANGE = (4,10) #32, 16 testar depois
#Tentar um histograma 

PEAK_SEARCH_WINDOW = 0.5

WHITEN = True
CENTER_ON_PEAK = True
PLOT_RESULTS = True

N_FREQ = 30
N_TIME = 40

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

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

print(f"\nStart time: {starttime}")
print(f"End time: {endtime}")

tr.plot(outfile=output_dir/"mseed_amplitude.pdf")

# ==========================================================
# STA/LTA
# ==========================================================

sta = 0.5
lta = 60
fmin = FRANGE[0]
fmax = FRANGE[1]

tr_original = tr.copy()
tr_band = tr_original.copy()
tr_band.filter("bandpass", freqmin=fmin, freqmax=fmax)

df = tr.stats.sampling_rate
cft = recursive_sta_lta(tr_band.data, int(sta*df), int(lta*df))

on_threshold = 20.0
off_threshold = 1.5

triggers = trigger_onset(cft, on_threshold, off_threshold)
print(f"\nNumber of triggers: {len(triggers)}")

# ==========================================================
# TRIGGER TIMES
# ==========================================================

trigger_times = []

for onset, offset in triggers:
    trigger_time = (tr.stats.starttime + onset/tr.stats.sampling_rate)
    trigger_times.append(trigger_time)

trigger_times = sorted(trigger_times)

if len(trigger_times)==0:

    print("No triggers found.")
    raise SystemExit

# ==========================================================
# ANALYSIS
# ==========================================================

results = []
all_q_data = []

for i, trigger_time in enumerate(trigger_times):

    center_time = trigger_time

    # ------------------------------------------------------
    # PEAK CENTERING
    # ------------------------------------------------------

    if CENTER_ON_PEAK:
        search_start = (trigger_time - PEAK_SEARCH_WINDOW)
        search_end = (trigger_time + PEAK_SEARCH_WINDOW)

        if (search_start >= tr.stats.starttime and search_end <= tr.stats.endtime):
            search_trace = tr.slice(starttime=search_start, endtime=search_end)

            if len(search_trace.data)>0:
                imax = np.argmax(np.abs(search_trace.data))
                center_time = (search_trace.stats.starttime + imax/search_trace.stats.sampling_rate)

    # ======================================================
    # WINDOW LOOP
    # ======================================================

    for half_width in WINDOWS:
        total_duration = 2*half_width
        if total_duration < 4./FRANGE[0]:
            continue
        center_idx = int((center_time-tr.stats.starttime)*df)
        nwin = int(half_width*df)
        i0 = center_idx-nwin
        i1 = center_idx+nwin

        if i0<0 or i1>=tr.stats.npts:
            continue

        data_event = tr.data[i0:i1]
        if len(data_event)==0:
            continue

        # ==================================================
        # GWPY TIME SERIES
        # ==================================================

        ts = TimeSeries(data_event.astype(np.float64), sample_rate=df, t0=0)

        # ==================================================
        # Q TRANSFORM
        # ==================================================

        try:
            qspec = ts.q_transform(frange=FRANGE, qrange=QRANGE, whiten=WHITEN)
            qspec.xindex = (qspec.xindex.value - half_width)

        except Exception as e:
            print("\nQ-transform failed:")
            print(e)

            continue

        # ==================================================
        # METRICS
        # ==================================================

        power = np.array(qspec.value)
        # Corrige orientação:
        # gwpy -> (tempo,frequência)
        # queremos -> (frequência,tempo)

        if power.shape[0] == len(qspec.xindex):
            power = power.T

        peak_energy = np.nanmax(power)
        mean_energy = np.nanmean(power)

        # ==================================================
        # EXTRACT DATA FOR CSV
        # ==================================================

        times_original = np.array(qspec.xindex.value)
        freqs_original = np.array(qspec.yindex.value)

        print(f"Power:{power.shape} | Freq:{len(freqs_original)} | Time:{len(times_original)}")

        try:
            interp = RegularGridInterpolator((freqs_original, times_original), power, bounds_error=False, fill_value=np.nan)
            freqs_new = np.linspace(freqs_original.min(), freqs_original.max(), N_FREQ)
            times_new = np.linspace(times_original.min(), times_original.max(), N_TIME)
            F,T = np.meshgrid(freqs_new, times_new, indexing='ij')
            points = np.column_stack([F.ravel(), T.ravel()])
            intensity = interp(points)
            for f,t,I in zip(F.ravel(),T.ravel(), intensity):
                all_q_data.append({"GPS_Trigger": trigger_time.timestamp, "Frequency_Hz": f, "Time_s": t, "Intensity": I})

        except Exception as e:
            print(f"Interpolation failed: {e}")

        results.append({"trigger_time":trigger_time, "center_time":center_time, "half_width":half_width, "peak_energy":peak_energy, "mean_energy": mean_energy})

        # ==================================================
        # PLOT
        # ==================================================

        if PLOT_RESULTS:
            fig = qspec.plot()
            ax = fig.axes[0]
            ax.set_title(f"Q-transform Window = ±{half_width} s\n Trigger = {center_time}")
            ax.set_xlabel("Time relative to trigger [s]")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_yscale("log")
            ax.set_ylim(FRANGE)
            ticks=[3,4,5,6,8,10,20,30]
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(i) for i in ticks])
            ax.set_xlim(-1,1)
            ax.axvline(0,color='red',linestyle='--',linewidth=1.5)
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.grid(False)
            mesh = ax.collections[0]
            mesh.set_edgecolors("face")
            mesh.set_antialiased(False)
            mesh.set_rasterized(True)
            cbar = fig.colorbar(mesh,ax=ax)
            cbar.set_label("Q-transform intensity")
            filename = (f"trigger_{i:04d}_window_{half_width}s.pdf")
            fig.savefig(output_dir/filename,dpi=300)
            plt.close()

# ==========================================================
# SUMMARY
# ==========================================================

summary_file = output_dir/"summary.txt"

def write_both(text,file):
    print(text)
    file.write(text+"\n")

with open(summary_file,"w") as f:
    write_both(f"Start time: {starttime}",f)
    write_both(f"End time: {endtime}",f)
    write_both(f"WINDOWS={WINDOWS}",f)
    write_both("="*60,f)
    for r in results:
        write_both(f"Trigger={r['trigger_time']} | Window=±{r['half_width']}s | Peak={r['peak_energy']:.4e} | Mean={r['mean_energy']:.4e}",f)

# ==========================================================
# SAVE GLOBAL CSV
# ==========================================================

df_all = pd.DataFrame(all_q_data)
csv_file = (output_dir/"qtransform_data.csv")
df_all.to_csv(csv_file,index=False)
print(f"\nCSV salvo:\n{csv_file}")