import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
from obspy.clients.fdsn import Client
from obspy import read


# ==========================================================
# WORKER FUNCTION
# ==========================================================

def process_chunk(args):

    (tr, inv, start,end, pre_filt, water_level) = args

    try:
        temp = tr.slice(starttime=start,endtime=end).copy()

        if len(temp.data) == 0:
            return None

        temp.remove_response(inventory=inv,output="VEL",pre_filt=pre_filt,water_level=water_level)

        return temp

    except Exception as e:

        print(f"Failed chunk: {start}")
        print(e)

        return None


# ==========================================================
# CLASS
# ==========================================================

class SeismicProcessor:

    def __init__(
        self,
        mseed_file,
        provider="INGV",
        xml_file="fdsn_station.xml",
        chunk_duration=6 * 3600,
        pre_filt=[0.001, 0.005, 40, 60],
        water_level=60,
        n_processes=None
    ):

        self.mseed_file = Path(mseed_file)

        self.provider = provider
        self.xml_file = xml_file

        self.chunk_duration = chunk_duration

        self.pre_filt = pre_filt
        self.water_level = water_level

        self.n_processes = (
            n_processes if n_processes
            else mp.cpu_count()
        )

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

        #print("\nMiniSEED loaded")
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

        self.inv.write(
            self.xml_file,
            format="STATIONXML",
            validate=True
        )

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

        #print("\nResampling complete")
        #print(f"Sampling rate: {self.tr.stats.sampling_rate} Hz")
        #print(f"Resample time: {time.time() - t0:.2f} s")

    # ======================================================
    # CREATE CHUNKS
    # ======================================================

    def create_tasks(self):
        tasks = []
        current = self.start

        while current < self.end:

            tasks.append(
                (
                    self.tr,
                    self.inv,
                    current,
                    current + self.chunk_duration,
                    self.pre_filt,
                    self.water_level
                )
            )
            current += self.chunk_duration
        return tasks

    # ======================================================
    # MULTIPROCESS RESPONSE REMOVAL
    # ======================================================

    def remove_response_parallel(self):
        print("\nStarting multiprocessing response removal...\n")
        t0 = time.time()
        tasks = self.create_tasks()

        with mp.Pool(self.n_processes) as pool:

            processed = pool.map(
                process_chunk,
                tasks
            )

        processed = [
            tr for tr in processed
            if tr is not None
        ]

        #print(f"\nProcessed chunks: {len(processed)}")
        #print(f"Parallel response removal time: " f"{time.time() - t0:.2f} s")

        return processed

    # ======================================================
    # RUN FULL PIPELINE
    # ======================================================

    def run(self):
        self.read_mseed()
        self.download_stationxml()
        self.resample()
        processed_stream = self.remove_response_parallel()
        return processed_stream


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    processor = SeismicProcessor(
        mseed_file="SENA-files/2021/eida_response_MN-SENA_20211201000000_20211231235959.mseed",
        chunk_duration=6 * 3600,
        n_processes=8
    )

    processed_stream = processor.run()
    print("\nProcessing complete!")