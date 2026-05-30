import time
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

xml_file = "fdsn_station.xml"
start_time = time.time()
client = Client("INGV")  # since your XML came from INGV

inv = client.get_stations(
    network="MN",
    station="SENA",
    starttime=UTCDateTime("2022-06-21"),
    endtime=UTCDateTime("2025-12-31"),
    level="response"   # <-- THIS is the key fix
)

inv.write(xml_file, format="STATIONXML", validate=True)




print(f"Write time: {time.time() - start_time} seconds")