from obspy.clients.fdsn import Client
from obspy import UTCDateTime

client = Client("INGV")  # since your XML came from INGV

inv = client.get_stations(
    network="MN",
    station="SENA",
    starttime=UTCDateTime("2022-06-21"),
    endtime=UTCDateTime("2025-12-31"),
    level="response"   # <-- THIS is the key fix
)

inv.write("fdsn_station.xml", format="STATIONXML")