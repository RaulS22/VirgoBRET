import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_inventory, UTCDateTime
from pathlib import Path

# TODO: Make this work =)

# =========================
# Paths
# =========================
input_dir = Path("/home/rauls/Desktop/VirgoBRET/SENA-files/2022")
inv = read_inventory("fdsn_station.xml")

# =========================
# Storage
# =========================
dates = []
daily_medians = []

# =========================
# Loop over files
# =========================
files = sorted(input_dir.glob("*.mseed"))

for file in files:
    print(f"Processing {file.name}")

    st = read(file)

    # Merge traces (handles fragmentation)
    st.merge(method=1, fill_value="interpolate")

    for tr in st:
        try:
            # Remove instrument response → velocity (m/s)
            tr.remove_response(inventory=inv, output="VEL", water_level=60)

            # Convert to nm/s
            data = tr.data * 1e9

            # Remove NaNs/Infs
            data = data[np.isfinite(data)]

            if len(data) == 0:
                continue

            # Median of absolute velocity
            median_val = np.median(np.abs(data))

            # Extract date
            date = tr.stats.starttime.date

            dates.append(date)
            daily_medians.append(median_val)

        except Exception as e:
            print(f"Error in {file.name}: {e}")
            continue

# =========================
# Aggregate by day (important!)
# =========================
from collections import defaultdict

daily_dict = defaultdict(list)

for d, val in zip(dates, daily_medians):
    daily_dict[d].append(val)

final_dates = []
final_medians = []

for d in sorted(daily_dict.keys()):
    final_dates.append(d)
    final_medians.append(np.median(daily_dict[d]))

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 5))
plt.plot(final_dates, final_medians, marker='o')
plt.xlabel("Date")
plt.ylabel("Median Ground Velocity (nm/s)")
plt.title("Daily Median Ground Velocity")
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()