import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy.clients.fdsn import Client
from obspy import read, read_inventory, Trace, UTCDateTime

# ============================================================
# Paths
# ============================================================

here = "/home/rauls/Desktop/VirgoBRET/Paralel"
mseed_file = "SENA-files/2021/eida_response_MN-SENA_20211201000000_20211231235959.mseed" 
xml_file = "fdsn_station.xml"

# ============================================================
# LOAD DATA
# ============================================================

print("Reading MiniSEED file...")
st = read(mseed_file)
tr = st[0]
print(tr)

# ============================================================
# DOWNLOAD RESPONSE ONLY ONCE
# ============================================================

try:
    print("Loading local StationXML...")
    inv = read_inventory(xml_file)

except:
    print("Downloading StationXML from INGV...")

    client = Client("INGV")

    inv = client.get_stations(
        network="MN",
        station="SENA",
        starttime=UTCDateTime("2021-06-21"),
        endtime=UTCDateTime("2025-12-31"),
        level="response"
    )

    inv.write(xml_file, format="STATIONXML", validate=True)

    print(f"StationXML saved to: {xml_file}")

# ============================================================
# OPTIONAL DECIMATION
# ============================================================

print("Original sampling rate:", tr.stats.sampling_rate)

# Reduce computational cost
if tr.stats.sampling_rate > 50:
    tr.decimate(2)

print("New sampling rate:", tr.stats.sampling_rate)

# ============================================================
# PROCESS IN CHUNKS
# ============================================================

chunk_duration = 600  # 10min windows

start = tr.stats.starttime
end   = tr.stats.endtime

current = start

processed_stream = []

print("\nStarting response removal...\n")

while current < end:

    print(f"Processing window: {current}")

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

print("\nResponse removal finished.")

# ============================================================
# MERGE RESULTS
# ============================================================

velocity_data = []
velocity_time = []

for tr in processed_stream:

    npts = tr.stats.npts
    dt = tr.stats.delta

    # Build absolute UTC datetime axis
    times = [
        (tr.stats.starttime + i * dt).datetime
        for i in range(npts)
    ]

    velocity_time.extend(times)
    velocity_data.extend(tr.data)

velocity_time = np.array(velocity_time)
velocity_data = np.array(velocity_data)

# ============================================================
# PLOT
# ============================================================

print("Generating plot...")

plt.figure(figsize=(18, 5))

plot_decimation = 100

plt.plot(
    velocity_time[::plot_decimation],
    velocity_data[::plot_decimation] * 1e9,
    linewidth=0.3
)

plt.title("Ground Velocity After Response Removal")
plt.xlabel("UTC Date")
plt.ylabel("Velocity (nm/s)")

plt.grid(True)
ax = plt.gca()

# Tick every 2 days
ax.xaxis.set_major_locator(
    mdates.DayLocator(interval=2)
)

# UTC formatting
ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%Y-%m-%d')
)
plt.xticks(rotation=45)
plt.tight_layout()
output_file = f"{here}/velocity_trace_chunked.pdf"
plt.savefig(output_file, dpi=300)
