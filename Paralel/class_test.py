import time
import numpy as np
from pathlib import Path
from obspy.clients.fdsn import Client
from obspy import read


class SeismicProcessor:
    """
    Class for:
        - Reading MiniSEED files
        - Downloading StationXML
        - Resampling
        - Removing instrument response in chunks
    """

    def __init__(
        self,
        mseed_file,
        provider="INGV",
        xml_file="fdsn_station.xml",
        chunk_duration=3600,
        pre_filt=[0.001, 0.005, 40, 60],
        water_level=60,
    ):

        self.mseed_file = Path(mseed_file)
        self.provider = provider
        self.xml_file = xml_file

        self.chunk_duration = chunk_duration
        self.pre_filt = pre_filt
        self.water_level = water_level

        self.client = Client(provider)

        self.st = None
        self.tr = None
        self.inv = None

        self.network = None
        self.station = None
        self.start = None
        self.end = None

    # ======================================================
    # READ DATA
    # ======================================================

    def read_mseed(self):
        t0 = time.time()

        self.st = read(str(self.mseed_file))
        self.tr = self.st[0]

        self.network = self.tr.stats.network
        self.station = self.tr.stats.station
        self.start = self.tr.stats.starttime
        self.end = self.tr.stats.endtime

        #print(f"\nMiniSEED loaded")
        #print(f"Network : {self.network}")
        #print(f"Station : {self.station}")
        #print(f"Start   : {self.start}")
        #print(f"End     : {self.end}")

        #print(f"Read time: {time.time() - t0:.2f} s")

    # ======================================================
    # DOWNLOAD XML
    # ======================================================

    def download_stationxml(self):
        t0 = time.time()

        self.inv = self.client.get_stations(
            network=self.network,
            station=self.station,
            starttime=self.start,
            endtime=self.end,
            level="response"
        )

        self.inv.write(self.xml_file,format="STATIONXML",validate=True)

        #print(f"\nStationXML saved -> {self.xml_file}")
        #print(f"Download time: {time.time() - t0:.2f} s")

    # ======================================================
    # RESAMPLE
    # ======================================================

    def resample(self, max_frequency=60):

        t0 = time.time()
        if self.tr.stats.sampling_rate > max_frequency:
            self.tr.decimate(2)

        self.tr.data = self.tr.data.astype(np.float32)

        #print(f"\nResampling complete")
        #print(f"New sampling rate: {self.tr.stats.sampling_rate} Hz")
        #print(f"Resample time: {time.time() - t0:.2f} s")

    # ======================================================
    # RESPONSE REMOVAL
    # ======================================================

    def remove_response_chunks(self):
        t0 = time.time()
        current = self.start
        processed = []

        print("\nStarting response removal...\n")

        while current < self.end:

            temp = self.tr.copy().trim(
                starttime=current,
                endtime=current + self.chunk_duration
            )

            if len(temp.data) == 0:
                current += self.chunk_duration
                continue

            try:

                temp.remove_response(
                    inventory=self.inv,
                    output="VEL",
                    pre_filt=self.pre_filt,
                    water_level=self.water_level
                )

                processed.append(temp)

                print(f"Processed: {current}")

            except Exception as e:

                print(f"Failed at {current}")
                print(e)

            current += self.chunk_duration

        #print(f"\nResponse removal time: {time.time() - t0:.2f} s")
        return processed

    # ======================================================
    # FULL PIPELINE
    # ======================================================

    def run(self):
        self.read_mseed()
        self.download_stationxml()
        self.resample()
        processed_stream = self.remove_response_chunks()
        return processed_stream


if __name__ == "__main__":
    processor = SeismicProcessor(
        mseed_file="SENA-files/2021/eida_response_MN-SENA_20211201000000_20211231235959.mseed"
    )

    processed_stream = processor.run()