import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
from obspy.clients.fdsn import Client
from obspy import read, Stream


# ==========================================================
# WORKER FUNCTION
# ==========================================================

def process_chunk(args):
    (tr, inv, start, end, pre_filt, water_level) = args

    try:
        temp = tr.slice(starttime=start,endtime=end).copy()

        if len(temp.data) == 0:
            return None

        temp.remove_response(
            inventory=inv,
            output="VEL",
            pre_filt=pre_filt,
            water_level=water_level
        )

        return temp

    except Exception as e:
        print(f"\nFailed chunk: {start}")
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
            n_processes
            if n_processes
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
        #print(f"\nRead time: {time.time() - t0:.2f} s")

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

    def resample(self, max_frequency=6):
        t0 = time.time()
        if self.tr.stats.sampling_rate > max_frequency:
            factor = int(self.tr.stats.sampling_rate / max_frequency)
            self.tr.decimate(factor)
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
            chunk_end = min(current + self.chunk_duration,self.end)
            tasks.append((self.tr, self.inv, current, chunk_end, self.pre_filt, self.water_level))
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
            processed = pool.map(process_chunk,tasks)

        processed = [tr for tr in processed if tr is not None]
        print(f"\nProcessed chunks: {len(processed)}")
        print("Parallel response removal time: " f"{time.time() - t0:.2f} s")
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

    # ======================================================
    # VALIDATE TIMES
    # ======================================================

    @staticmethod
    def validate_times(raw_file, processed_file):
        raw_st = read(raw_file)
        proc_st = read(processed_file)
        raw_tr = raw_st[0]
        proc_tr = proc_st[0]

        print("\n=== TIME CHECK ===")
        print(f"RAW START : {raw_tr.stats.starttime}")
        print(f"PROC START: {proc_tr.stats.starttime}")
        print(f"RAW END   : {raw_tr.stats.endtime}")
        print(f"PROC END  : {proc_tr.stats.endtime}")

        start_match = (raw_tr.stats.starttime== proc_tr.stats.starttime)
        end_match = (raw_tr.stats.endtime== proc_tr.stats.endtime)

        if start_match and end_match:
            print("\nPASS: Start and end times match.")
        else:
            print("\nFAIL: Time mismatch detected.")

        return start_match and end_match

    # ======================================================
    # VALIDATE FULL TRACE
    # ======================================================

    @staticmethod
    def validate_trace(raw_file, processed_file):
        raw = read(raw_file)[0]
        proc = read(processed_file)[0]
        checks = {
            "starttime":raw.stats.starttime== proc.stats.starttime,
            "endtime":raw.stats.endtime== proc.stats.endtime,
            "sampling_rate":raw.stats.sampling_rate== proc.stats.sampling_rate,
            "npts":raw.stats.npts== proc.stats.npts,
            }

        print("\n=== TRACE VALIDATION ===")

        for key, value in checks.items():
            status = "PASS" if value else "FAIL"
            print(f"{key}: {status}")
        return all(checks.values())


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    input_file = Path("SENA-files/2022/eida_response_MN-SENA_20220101000000_20220131235959.mseed")
    output_file = "processed_output.mseed"

    processor = SeismicProcessor(
        mseed_file=input_file,
        chunk_duration=31*86399,
        n_processes=8
    )

    processed_stream = processor.run()
    final_stream = Stream(traces=processed_stream)
    final_stream.write(output_file,format="MSEED")
    print("\nProcessing complete!")

    SeismicProcessor.validate_times(input_file, output_file)
    SeismicProcessor.validate_trace(input_file, output_file)