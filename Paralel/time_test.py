import os
import time
import numpy as np
from obspy.clients.fdsn import Client
from obspy import read, UTCDateTime
from pathlib import Path

# ==========================================================
# INPUT
# ==========================================================

start_time = time.time()
mseed_file = "SENA-files/2021/eida_response_MN-SENA_20211201000000_20211231235959.mseed"
client = Client("INGV")

# ==========================================================
# READ MSEED
# ==========================================================
st = read(mseed_file)

# Assuming all traces belong to the same station
tr = st[0]
network = tr.stats.network
station = tr.stats.station
start = tr.stats.starttime
end = tr.stats.endtime

#print(f"Network : {network}")
#print(f"Station : {station}")
#print(f"Start   : {start}")
#print(f"End     : {end}")

#print(f"Read time: {time.time() - start_time} seconds")


# ==========================================================
# CREATE STATIONXML
# ==========================================================

start_time = time.time()
xml_file = "fdsn_station.xml"
client = Client("INGV")  # since your XML came from INGV

inv = client.get_stations(
    network="MN",
    station="SENA",
    starttime=start,
    endtime=end,
    level="response"  
)

inv.write(xml_file, format="STATIONXML", validate=True)
#print(f"Write .xml time: {time.time() - start_time} seconds")

# ==========================================================
# RESAMPLE AND CONVERT TO FLOAT32
# ==========================================================

start_time = time.time()
#print("Starting resample")

if tr.stats.sampling_rate > 60:
    tr.decimate(2)
#print(f"Resample time: {time.time() - start_time} seconds")

#start_time = time.time()
velocity = tr.data.astype(np.float32)
fs = tr.stats.sampling_rate
dt = tr.stats.delta
starttime = tr.stats.starttime.datetime

#print(f"Conversion time: {time.time() - start_time} seconds")

# ==========================================================
# RESPONSE REMOVAL IN CHUNCKS
# ==========================================================

start_time = time.time()
chunk_duration = 3600  # 1h windows
current = start
processed_stream = []
print("\nStarting response removal...\n")

while current < end:
    #print(f"Processing window: {current}")
    temp = tr.copy().trim(
        starttime=current,
        endtime=current + chunk_duration
    )

    # Skip empty chunks
    if len(temp.data) == 0:
        current += chunk_duration
        continue

    try:
        temp.remove_response(
            inventory=inv,
            output="VEL",
            pre_filt=[0.001, 0.005, 40, 60], #band limitation
            water_level=60 #numerical floor for deconvolution stability
        )
        processed_stream.append(temp)

    except Exception as e:
        print(f"Failed at {current}")
        print(e)

    current += chunk_duration

#print(f"Response removal time: {time.time() - start_time} seconds")
#Converting to float32 saves ~1min for the first test