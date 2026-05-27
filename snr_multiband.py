import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from obspy import read
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset, z_detect


# ==========================================================
# PREPROCESSING
# ==========================================================

def preprocess_trace(tr, inventory):
    tr = tr.copy()
    tr.detrend("demean")
    tr.detrend("linear")

    # taper can sometimes hurt SNR
    # tr.taper(max_percentage=0.05)

    tr.remove_response(inventory=inventory,output="VEL",water_level=20)
    return tr


# ==========================================================
# BAND FILTERING
# ==========================================================

def create_band_traces(tr, bands):
    traces = {}
    for fmin, fmax in bands:
        key = f"{fmin}-{fmax}Hz"
        temp = tr.copy()
        temp.filter("bandpass",freqmin=fmin,freqmax=fmax)

        traces[key] = temp
    return traces


# ==========================================================
# STA/LTA DETECTOR
# ==========================================================

def run_sta_lta(tr,sta=5,lta=600,on_threshold=2.5,off_threshold=1.5):
    df = tr.stats.sampling_rate
    cft = recursive_sta_lta(tr.data,int(sta * df),int(lta * df))
    triggers = trigger_onset(cft,on_threshold,off_threshold)

    return cft, triggers


# ==========================================================
# Z-DETECT
# ==========================================================

def run_z_detect(tr,window=10,on_threshold=2.5,off_threshold=1.5):
    df = tr.stats.sampling_rate
    cft = z_detect(tr.data,int(window * df))
    triggers = trigger_onset(cft,on_threshold,off_threshold)
    return cft, triggers


# ==========================================================
# REMOVE TRIGGERS TOO CLOSE
# ==========================================================

def filter_trigger_times(trigger_times,min_separation=5):
    filtered = []
    last_time = None

    for t in trigger_times:
        if last_time is None:
            filtered.append(t)
            last_time = t

        elif (t - last_time) > min_separation:
            filtered.append(t)
            last_time = t
    return filtered


# ==========================================================
# EXTRACT TRIGGER TIMES
# ==========================================================

def extract_trigger_times(tr,triggers,ignore_first_seconds=0):

    df = tr.stats.sampling_rate
    trigger_times = []
    for trig in triggers:
        sample = trig[0]
        t = tr.stats.starttime + sample / df
        # ignore initial noisy section
        if (t - tr.stats.starttime) < ignore_first_seconds:
            continue

        trigger_times.append(t)
    return trigger_times


# ==========================================================
# PLOTTING
# ==========================================================

def plot_detection(
    tr,cft,trigger_times,detector_name,band_name,output_dir="plots_snr"):
    Path(output_dir).mkdir(exist_ok=True)
    time_array = tr.times("matplotlib")
    fig, ax = plt.subplots(2,1,figsize=(14, 8),sharex=True)

    # ------------------------------------------------------
    # WAVEFORM
    # ------------------------------------------------------

    ax[0].plot(time_array,tr.data * 1e9,linewidth=0.8)

    for pt in trigger_times:
        ax[0].axvline(pt.datetime,color="r",linestyle="--",alpha=0.3)

    ax[0].set_ylabel("Velocity [nm/s]")
    ax[0].set_title(f"{detector_name} | {band_name}")
    ax[0].grid(alpha=0.3)

    # ------------------------------------------------------
    # CHARACTERISTIC FUNCTION
    # ------------------------------------------------------

    ax[1].plot(time_array,cft,linewidth=0.8)
    ax[1].set_ylabel("Characteristic Function")
    ax[1].grid(alpha=0.3)
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))

    fig.autofmt_xdate()
    filename = (f"{output_dir}/"f"{detector_name}_{band_name}.pdf")
    plt.savefig(filename,bbox_inches="tight")
    plt.close()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    st = read("14-08-25-Fabi.mseed")
    tr = st[0]
    starttime = tr.stats.starttime
    endtime = tr.stats.endtime

    # ------------------------------------------------------
    # DOWNLOAD INVENTORY
    # ------------------------------------------------------

    client = Client("INGV")

    inv = client.get_stations(network="MN",station="SENA",starttime=starttime,endtime=endtime,level="response")
    inv.write("fdsn_station.xml",format="STATIONXML",validate=True)

    # ------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------

    print("Preprocessing trace...")
    tr = preprocess_trace(tr, inv)
    bands = [(0.03, 0.1),(0.1, 0.3),(0.3, 1.0),(1.0, 3.0)]

    print("Creating filtered traces...")
    band_traces = create_band_traces(tr,bands)

    methods = {"STA_LTA": run_sta_lta,"Z_DETECT": run_z_detect}

    for band_name, tr_band in band_traces.items():
        print("\n" + "=" * 60)
        print(f"Processing band: {band_name}")

        for detector_name, detector_function in methods.items():
            print(f"Running method: {detector_name}")
            # run detector
            cft, triggers = detector_function(tr_band)

            # convert trigger indices to UTC times
            trigger_times = extract_trigger_times(tr_band,triggers,ignore_first_seconds=1200)

            # remove nearby triggers
            trigger_times = filter_trigger_times(trigger_times,min_separation=20)

            print(f"Number of triggers: "f"{len(trigger_times)}")

            # print trigger times
            # for t in trigger_times:print(t)
            plot_detection(tr_band,cft,trigger_times,detector_name,band_name)

    print("\nDone.")