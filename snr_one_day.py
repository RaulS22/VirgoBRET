import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import read, read_inventory
from obspy.clients.fdsn import Client
from pathlib import Path
from obspy.signal.trigger import recursive_sta_lta, trigger_onset


st = read("22-02-25-Raul.mseed")
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

#Aqui e feito um taper para suavizar as "bordas" referentes a 5% em cada extremo
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

tr.filter("bandpass", freqmin=0.03, freqmax=1.0)
df = tr.stats.sampling_rate

#Plot this band velocity
tr.detrend("demean")
tr.detrend("linear")
tr.remove_response(inventory=inv, output="VEL", water_level=20) #velcidade em m/s 
plt.figure(figsize=(12,5))
plt.plot(time_array, tr.data*1e9)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
plt.gcf().autofmt_xdate()
plt.xlabel("Time [s]")
plt.ylabel("Velocity [nm/s]")
plt.grid()
plt.savefig("raul_teste_velocity_band_limited.pdf")


# STA/LTA windows in seconds
sta, lta = 5, 600

cft = recursive_sta_lta(tr.data, int(sta * df), int(lta * df)) # func caracteristica
on_threshold, off_threshold = 2.5, 1.5 
triggers = trigger_onset(cft, on_threshold, off_threshold) #deteccao dos triggers
print(f"Number of triggers: {len(triggers)}")

#partindo dos tempos (s) dos tiggers obter o tempo (UTC)
p_times = []

for trig in triggers:
    p_sample = trig[0]
    p_time = tr.stats.starttime + 1200 + p_sample / df #tira os 20 primeiros minutos porque isso atrapalha muito o SNR
    p_times.append(p_time)
    print(p_time)


min_separation = 20
filtered_p_times = []
last_time = None

# excluir picos muito próximos
for trig in triggers:
    pt = tr.stats.starttime + trig[0] / df
    if last_time is None or (pt - last_time) > min_separation:
        filtered_p_times.append(pt)
        last_time = pt

fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True)
ax[0].plot(time_array, tr.data * 1e9, linewidth=0.8)
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

