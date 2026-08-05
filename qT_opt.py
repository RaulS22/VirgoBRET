import numpy as np
import matplotlib.pyplot as plt
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

MSEED_FILE = "SENA-files/2025/eida_response_MN-SENA_20250201000000_20250228235959.mseed" 

STA = 0.5
LTA = 60
ON_THRESHOLD = 30 #20 was used before
OFF_THRESHOLD = 7.5 #1.5 was used before

FRANGE = (3,30)
QRANGE = (64,128)
WHITEN = True

FREQ_BINS = 27
TIME_BINS = 41

PERPLEXITY = 30
DIMENSIONALITY = 2

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
    return triggers, tr
    

triggers, tr = seismic_trig(MSEED_FILE, FRANGE[0], FRANGE[1], UTC=True, p=True)
#print(triggers)

#######################################################################
# Q-transform  
#######################################################################

Rever




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

    print(f"File:{MSEED_FILE}")
    print(f"Start time: {starttime} | End time: {endtime}")
    tr.plot(outfile=f"{output_dir}/mseed_amplitude.pdf")

    def write_both(text, file):
        print(text)
        file.write(text + "\n")

    with open(summary_file, "w") as f:
        write_both(f"Start time: {starttime} \nEnd time: {endtime}", f)
        write_both(f"Inputs: FRANGE = {FRANGE}, QRANGE = {QRANGE}, WHITHEN = {WHITEN}", f)
        write_both(f"Parameters: sta = {STA}, lta = {LTA}, threshold = {ON_THRESHOLD}", f)
        write_both(f"Images: frequency_bins = {FREQ_BINS}, time_bins = {TIME_BINS}", f)

    fig = qspec.plot()
    ax = fig.axes[0]
    ax.set_title(f"Q-transform Window = {2*half_width} s\n Trigger = {center_time}")
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
