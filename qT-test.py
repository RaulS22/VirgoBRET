import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from pathlib import Path
from gwpy.timeseries import TimeSeries

#TODO: Understand the qTransform
#TODO: Check if the peals are actually peaks
#TODO: Solve the date problem

# ==========================================================
# USER INPUTS
# ==========================================================

MSEED_FILE = "22-02-25-Raul.mseed"

WINDOWS = [1, 2, 5, 10, 20, 30, 40]      # seconds on each side
FRANGE = (5, 10)
fmin, fmax = FRANGE[0], FRANGE[1]
QRANGE = (4, 20)

OUTPUT_DIR = Path("qTransform")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# READ DATA
# ==========================================================
st = read(MSEED_FILE)
tr = st[0]
starttime = tr.stats.starttime
endtime = tr.stats.endtime

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

sta = 5
lta = 600

tr_band = tr.copy()
#tr_band.filter("bandpass", freqmin=fmin, freqmax=fmax)
df = tr_band.stats.sampling_rate

cft = recursive_sta_lta(tr_band.data, int(sta * df), int(lta * df))
on_threshold = 5.0
off_threshold = 1.5

triggers = trigger_onset(cft, on_threshold, off_threshold)
print(f"\nNumber of triggers: {len(triggers)}")

# ==========================================================
# CONVERT TRIGGERS TO UTCDateTime
# ==========================================================

trigger_times = []

for onset, offset in triggers:
    trigger_time = (tr.stats.starttime +onset / tr.stats.sampling_rate)
    trigger_times.append(trigger_time)
    #print(f"Trigger: {trigger_time} (sample {onset})")

if len(trigger_times) == 0:
    print("No triggers found.")
    raise SystemExit

#print(trigger_times)
##

for ntrigger, t0 in enumerate(trigger_times):
    print(f"\nProcessing trigger {ntrigger+1}/{len(trigger_times)}")
    for w in WINDOWS:
        t1 = t0 - w/2
        t2 = t0 + w/2
        tr_cut = tr_band.slice(starttime=t1,endtime=t2).copy()
        if len(tr_cut.data) == 0:
            continue

        #print(tr.stats.starttime)
        data = TimeSeries(tr_cut.data,sample_rate=tr_cut.stats.sampling_rate,t0=0)
        #data = TimeSeries(tr_cut.data,sample_rate=tr_cut.stats.sampling_rate,t0=tr_cut.stats.starttime.timestamp)
        #print(data)
        #print(UTCDateTime(t0))
        q = data.q_transform(whiten=True)
        q.plot()
        #q = data.q_transform(frange=FRANGE,qrange=QRANGE, whiten=False)
        #plt.show()
        filename = (f"trigger_{ntrigger+1}_window_{w}s.pdf")
        #filename = (f"trigger_{ntrigger+1}.pdf")
        plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
print("\nDone.")
    
