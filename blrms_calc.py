import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client

# Initial definitions 
bands = [(0.03, 0.1),(0.1, 0.3),(0.3, 1.0),(1.0, 3.0)]
window = 60 # the blrms will be calculated for each minute


# Reading and processing of mseed file 
st = read("22-02-25-Raul.mseed")
tr = st[0]
starttime = tr.stats.starttime
endtime = tr.stats.endtime

client = Client("INGV")
inv = client.get_stations(network="MN",station="SENA",starttime=starttime,endtime=endtime,level="response")
inv.write("fdsn_station.xml",format="STATIONXML",validate=True)

def preprocess_trace(tr, inventory):
    tr = tr.copy()
    tr.detrend("demean")
    tr.detrend("linear")

    # taper can sometimes hurt SNR
    # tr.taper(max_percentage=0.05)

    tr.remove_response(inventory=inventory,output="VEL",water_level=20)
    return tr

tr = preprocess_trace(tr, inv)

# BLRMS

def compute_blrms(tr, bands, window):
    fs = tr.stats.sampling_rate
    nwin = int(window * fs)
    results = {}

    for fmin, fmax in bands:
        tr_band = tr.copy()
        tr_band.filter("bandpass",freqmin=fmin,freqmax=fmax)
        data = tr_band.data
        rms_values = []
        times = []

        for i in range(0, len(data) - nwin, nwin):
            segment = data[i:i+nwin]
            rms = np.sqrt(np.mean(segment**2))
            rms_values.append(rms)
            times.append(tr.stats.starttime + i/fs + window/2)

        results[(fmin, fmax)] = {"times": np.array(times),"blrms": np.array(rms_values)}
    return results

blrms = compute_blrms(tr,bands=bands,window=window)
#print(blrms)

# Atempt to only use the data between the 1st and the 3rd quartiles
#for band, result in blrms.items():
#    blrms_values = result["blrms"]
#    Q1 = np.percentile(blrms_values, 25)
#    Q3 = np.percentile(blrms_values, 75)
#    mask = (blrms_values >= Q1) & (blrms_values <= Q3)
#    result["times_q1-q3"] = result["times"][mask]
#    result["blrms_q1-q3"] = blrms_values[mask]

# Plot
fig, ax = plt.subplots(figsize=(12,6))
for band, result in blrms.items():
    times = [t.datetime for t in result["times"]]
    ax.plot(times,result["blrms"]*1e9,label=f"{band[0]}-{band[1]} Hz")

ax.set_ylabel("Velocity RMS [nm/s]")
ax.set_xlabel("Time")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
plt.savefig("BLRMS_Test.pdf")