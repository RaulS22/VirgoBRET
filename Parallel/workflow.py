# MiniSEED -> Response removal (parallel) -> Corrected velocity -> Band filtering (parallel by band) -> Window RMS -> Daily median -> Plot/save

import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict

from obspy import read, Stream
from obspy.clients.fdsn import Client

from scipy.signal import butter, sosfiltfilt


# TODO: fix this

# ==========================================================
# RESPONSE REMOVAL WORKER
# ==========================================================

def response_worker(args):

    (tr, inv, start, end, pre_filt, water_level) = args

    try:
        temp = tr.slice(starttime=start, endtime=end).copy()

        if len(temp.data) == 0:
            return None

        temp.remove_response(
            inventory=inv,
            output="VEL",
            pre_filt=pre_filt,
            water_level=water_level
        )

        temp.data = temp.data.astype(np.float32)
        return temp

    except Exception as e:
        print(f"Failed response removal: {start}")
        print(e)
        return None


# ==========================================================
# STORAGE CLASS
# ==========================================================

class IntermediateStorage:
    def __init__(self, base_directory="Products"):
        self.base = Path(base_directory)
        self.base.mkdir(parents=True, exist_ok=True)
        self.velocity_dir = (self.base / "Velocity")
        self.blrms_dir = (self.base / "BLRMS")
        self.velocity_dir.mkdir(exist_ok=True)
        self.blrms_dir.mkdir(exist_ok=True)

    # ======================================================

    def save_velocity_trace(self,tr,filename):
        path = (self.velocity_dir / filename)

        st = Stream([tr])
        st.write(str(path), format="MSEED")
        print(f"\nSaved velocity:")
        print(path)

    # ======================================================

    def load_velocity_trace(self,filename):

        path = (self.velocity_dir / filename)
        st = read(str(path))
        tr = st[0]

        print(f"\nLoaded velocity:")
        print(path)
        return tr

    # ======================================================

    def save_blrms(self,results):
        for band_name, (times, values) in results.items():
            path = (self.blrms_dir /f"{band_name}.npz")

            np.savez(path,times=np.array(times,dtype=str),values=np.array(values))

            print(f"\nSaved BLRMS:")
            print(path)

    # ======================================================

    def load_blrms(self):
        results = {}
        files = sorted(self.blrms_dir.glob("*.npz"))

        for file in files:
            data = np.load(file)
            band_name = file.stem
            results[band_name] = (data["times"], data["values"])
            print(f"\nLoaded BLRMS:")
            print(file)

        return results


# ==========================================================
# RESPONSE REMOVAL CLASS
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
        n_processes=8
    ):

        self.mseed_file = Path(mseed_file)
        self.provider = provider
        self.xml_file = xml_file
        self.chunk_duration = chunk_duration
        self.pre_filt = pre_filt
        self.water_level = water_level
        self.n_processes = n_processes
        self.client = Client(provider)
        self.tr = None
        self.inv = None

    # ======================================================

    def read_data(self):
        print("\nReading MiniSEED...\n")
        st = read(str(self.mseed_file))
        self.tr = st[0]
        print(self.tr)

    # ======================================================

    def download_inventory(self):
        tr = self.tr
        print("\nDownloading StationXML...\n")

        self.inv = self.client.get_stations(
            network=tr.stats.network,
            station=tr.stats.station,
            starttime=tr.stats.starttime,
            endtime=tr.stats.endtime,
            level="response"
        )

        self.inv.write(
            self.xml_file,
            format="STATIONXML"
        )

    # ======================================================

    def preprocess(self):
        tr = self.tr
        tr.detrend("demean")
        tr.detrend("linear")
        tr.taper(0.05)
        if tr.stats.sampling_rate > 60:
            tr.decimate(2)

    # ======================================================

    def create_tasks(self):
        tr = self.tr
        start = tr.stats.starttime
        end = tr.stats.endtime
        current = start
        tasks = []
        while current < end:
            tasks.append((tr, self.inv, current, current + self.chunk_duration, self.pre_filt, self.water_level))
            current += self.chunk_duration
        return tasks

    # ======================================================

    def remove_response_parallel(self):
        print("\nStarting response removal...\n")
        t0 = time.time()
        tasks = self.create_tasks()
        with mp.Pool(self.n_processes) as pool:
            processed = pool.map(response_worker,tasks)

        processed = [tr for tr in processed if tr is not None]
        st = Stream(processed)
        st.merge(method=1)

        print(f"\nResponse removal finished " f"in {time.time() - t0:.2f} s")
        return st[0]

    # ======================================================

    def run(self):
        self.read_data()
        self.download_inventory()
        self.preprocess()
        velocity_trace = (self.remove_response_parallel())
        return velocity_trace


# ==========================================================
# BLRMS WORKER
# ==========================================================

def blrms_worker(args):
    (tr, band_name, fmin, fmax, window) = args
    print(f"\nProcessing band: {band_name}")
    fs = tr.stats.sampling_rate
    data = tr.data.astype(np.float32)

    # ======================================================
    # BANDPASS FILTER
    # ======================================================

    sos = butter(4, [fmin, fmax], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos,data)

    # ======================================================
    # RMS WINDOWS
    # ======================================================

    nwin = int(window * fs)
    rms_values = []
    rms_times = []

    starttime = tr.stats.starttime
    for i in range(0, len(filtered) - nwin, nwin):
        chunk = filtered[i:i + nwin]
        rms = np.sqrt(np.mean(chunk**2))
        rms_values.append(rms * 1e9)  # m/s -> nm/s
        rms_times.append((starttime + i / fs).datetime)

    # ======================================================
    # DAILY MEDIAN
    # ======================================================

    daily = defaultdict(list)

    for t, v in zip(rms_times, rms_values):
        daily[t.date()].append(v)

    daily_times = []
    daily_medians = []

    for day in sorted(daily.keys()):
        daily_times.append(day)
        daily_medians.append(np.median(daily[day]))

    return (band_name,daily_times,daily_medians)

# ==========================================================
# BLRMS CLASS
# ==========================================================

class ParallelBLRMS:

    def __init__(
        self,
        velocity_trace,
        bands,
        window=60,
        n_processes=4
    ):

        self.tr = velocity_trace
        self.bands = bands
        self.window = window
        self.n_processes = n_processes
        self.results = {}

    # ======================================================

    def create_tasks(self):
        tasks = []
        for band_name, (fmin,fmax) in self.bands.items():
            tasks.append((self.tr, band_name, fmin, fmax, self.window))

        return tasks

    # ======================================================

    def compute(self):
        print("\nStarting BLRMS...\n")

        t0 = time.time()
        tasks = self.create_tasks()

        with mp.Pool(self.n_processes) as pool:

            results = pool.map(blrms_worker, tasks)

        for band_name, times, values in results:
            self.results[band_name] = (times, values)

        print(f"\nBLRMS finished " f"in {time.time() - t0:.2f} s")

    # ======================================================

    def plot(self):

        plt.figure(figsize=(12, 5))

        for band_name, (times,values) in self.results.items():
            plt.plot(
                times,
                values,
                marker="o",
                label=band_name
            )

        plt.yscale("log")
        plt.xlabel("Date")
        plt.ylabel("Daily Median Ground Velocity [nm/s]")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("BLRMs.pdf", dpi=300)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    storage = IntermediateStorage()

    RUN_RESPONSE_REMOVAL = False
    if RUN_RESPONSE_REMOVAL:

        processor = SeismicProcessor(
            mseed_file="SENA-files/2021/eida_response_MN-SENA_20211201000000_20211231235959.mseed",
            chunk_duration=6 * 3600,
            n_processes=8
        )

        velocity_trace = processor.run()

        # --------------------------------------------------
        # SAVE INTERMEDIATE PRODUCT
        # --------------------------------------------------

        storage.save_velocity_trace(velocity_trace, filename="SENA_2021_velocity.mseed")

    else:
        velocity_trace = (storage.load_velocity_trace("SENA_2021_velocity.mseed"))

    # ======================================================
    # BLRMS BANDS
    # ======================================================

    bands = {"microseism": (0.1, 1), "anthropogenic": (1, 3), "cultural": (3, 10), "high_frequency": (10, 30)}

    # ======================================================
    # BLRMS
    # ======================================================

    blrms = ParallelBLRMS(velocity_trace=velocity_trace, bands=bands, window=60, n_processes=8)
    blrms.compute()
    storage.save_blrms(blrms.results)
    blrms.plot()