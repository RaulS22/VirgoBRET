from obspy import read
import matplotlib.pyplot as plt

st = read("SENA-files/2022/eida_response_MN-SENA_20220101000000_20220131235959.mseed")
tr = st[0]
times = tr.times()
velocity = tr.data

# Plot
plt.figure(figsize=(12, 4))
plt.plot(times, velocity, linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.title(f"{tr.id} Velocity Trace")
plt.grid(True)

plt.tight_layout()
plt.savefig("velocity_jan_22.pdf", dpi=300)