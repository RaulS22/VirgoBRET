import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import read, read_inventory
from obspy.clients.fdsn import Client
from pathlib import Path
from obspy.signal.trigger import recursive_sta_lta, trigger_onset, z_detect, ar_pick


st = read("14-08-25-Fabi.mseed")
tr = st[0]
starttime = tr.stats.starttime
endtime = tr.stats.endtime

xml_file = "fdsn_station.xml"
client = Client("INGV") 

inv = client.get_stations(
    network="MN",
    station="SENA",
    starttime=starttime,
    endtime=endtime,
    level="response" 
)

inv.write(xml_file, format="STATIONXML", validate=True)

# Aqui e feito um taper para suavizar as "bordas" referentes a 5% em cada extremo
tr.detrend("demean")
tr.detrend("linear")
#tr.taper(max_percentage=0.05) #atrapalha pra dedeu no SNR
tr.remove_response(inventory=inv, output="VEL", water_level=20) #velcidade em m/s 
#o wtater_level e responsevel por suavizar os vales, diminuir aumenta o tempo de execucao, aumentar mata muita informacao

#time_array = tr.times() #time in s 
#time_array = np.array([starttime.datetime + np.timedelta64(int(t * 1e6), 'us') for t in tr.times()], dtype='datetime64[us]') #tempo para UTC
time_array = tr.times("matplotlib")

# #Plot
# plt.figure(figsize=(12,5))
# plt.plot(time_array, tr.data*1e9)
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
# plt.gcf().autofmt_xdate()
# plt.xlabel("Time [s]")
# plt.ylabel("Velocity [nm/s]")
# #plt.title("Ground Velocity")
# plt.grid()
# plt.savefig("raul_teste_velocity.pdf")

# Tentativa de cálculo de SNR a partir dos p-peaks
# Importante discutir depois o que de fato esta sendo considerado ruido e o que nao
# LTA -> Long term average; STA -> Short term average

# Para mais de uma banda
df = tr.stats.sampling_rate
tr_original = tr.copy() #importante criar a copia

bands = [(0.03, 0.1), (0.1, 1.0), (1.0, 3.0)]
filtered_traces = []

for fmin, fmax in bands:
    tr_band = tr_original.copy()
    tr_band.filter("bandpass", freqmin=fmin, freqmax=fmax)
    filtered_traces.append(tr_band)
    #print(f"Created band: {fmin} - {fmax} Hz")
df = tr.stats.sampling_rate

tr_low  = filtered_traces[0]
tr_mid  = filtered_traces[1]
tr_high = filtered_traces[2]

# Aplicar pra cada um
tr_sta = tr_mid

# STA/LTA windows
sta, lta = 5, 600

cft = recursive_sta_lta(tr_sta.data, int(sta * df), int(lta * df))
on_threshold, off_threshold = 2.5, 1.5
triggers = trigger_onset(cft, on_threshold, off_threshold)
print(f"Number of triggers STA/LTA: {len(triggers)}")

# partindo dos tempos (s) dos tiggers obter o tempo (UTC)
p_times = []

for trig in triggers:
    p_sample = trig[0]
    p_time = tr_sta.stats.starttime + 1200 + p_sample / df #tira os 20 primeiros minutos porque isso atrapalha muito o SNR
    p_times.append(p_time)
    print(p_time)


min_separation = 20
filtered_p_times = []
last_time = None

# excluir picos muito próximos
for trig in triggers:
    pt = tr_sta.stats.starttime + trig[0] / df
    if last_time is None or (pt - last_time) > min_separation:
        filtered_p_times.append(pt)
        last_time = pt

fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
ax[0].plot(time_array, tr_sta.data*1e9, linewidth=0.8)
for pt in p_times:
    ax[0].axvline(pt.datetime, color='r', linestyle='--', alpha=0.3)
#ax[0].legend()
ax[0].set_ylabel("Velocity [nm/s]")

# STA/LTA characteristic function
ax[1].plot(time_array, cft, linewidth=0.8)
ax[1].axhline(on_threshold, color='g', linestyle='--')
ax[1].axhline(off_threshold, color='r', linestyle='--')
ax[1].set_ylabel("STA/LTA")

ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
fig.autofmt_xdate()
plt.savefig("SNR_STA_LTA.pdf")

# Z detect

tr_detect = tr_mid
z_window = 10
nsta = int(z_window * df)
cft = z_detect(tr_detect.data, nsta)
on_threshold, off_threshold = 2.5, 1.5
triggers = trigger_onset(cft, on_threshold, off_threshold)
print(f"Number of triggers Z-Detect: {len(triggers)}")

p_times = []
for trig in triggers:
    p_sample = trig[0]
    p_time = (tr_detect.stats.starttime + p_sample / df)
    p_times.append(p_time)
    print(p_time)

fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
ax[0].plot(time_array, tr_detect.data*1e9, linewidth=0.8)

for pt in p_times:
    ax[0].axvline(pt.datetime, color='r', linestyle='--', alpha=0.3)

ax[0].set_ylabel("Velocity [nm/s]")

ax[1].plot(time_array, cft, linewidth=0.8)
ax[1].axhline(on_threshold,color='g', linestyle='--')
ax[1].axhline(off_threshold, color='r', linestyle='--')
ax[1].set_ylabel("Z-Detect")
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
fig.autofmt_xdate()
plt.savefig("Z_DETECT.pdf")

# # AR pick (GPT)
# 
# tr_ar = tr_mid
# df = tr_ar.stats.sampling_rate
# 
# # Run AR picker
# p_pick, s_pick = ar_pick(
#     tr_ar.data,   # Z
#     tr_ar.data,   # N (placeholder)
#     tr_ar.data,   # E (placeholder)
#     df,
# 
#     # Bandpass for AR picker internally
#     0.1,              # f1
#     1.0,              # f2
# 
#     # STA/LTA windows
#     5.0,                # lta_p
#     600.0,              # sta_p
#     5.0,                # lta_s
#     600.0,              # sta_s
# 
#     # AR model parameters
#     3,                # m_p
#     3,                # m_s
# 
#     # Variance windows
#     10,              # l_p
#     10               # l_s
# )
# 
# #print("P arrival:", p_pick)
# #print("S arrival:", s_pick)
# 
# p_time = tr_ar.stats.starttime + p_pick
# s_time = tr_ar.stats.starttime + s_pick
# #print("P UTC:", p_time)
# #print("S UTC:", s_time)
# 
# # ==========================================
# # Plot
# # ==========================================
# 
# plt.figure(figsize=(12,5))
# plt.plot(time_array, tr_ar.data*1e9)
# plt.axvline(p_time.datetime, color='r', linestyle='--', label='P')
# plt.axvline(s_time.datetime, color='b', linestyle='--', label='S')
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
# plt.gcf().autofmt_xdate()
# plt.xlabel("Time")
# plt.ylabel("Velocity [nm/s]")
# plt.legend()
# plt.grid()
# plt.savefig("AR_PICK.pdf")