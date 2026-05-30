import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_inventory, UTCDateTime
from pathlib import Path
import gc
import re
from concurrent.futures import ProcessPoolExecutor


# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path("/home/rauls/Desktop/VirgoBRET/SENA-files/2022")
INVENTORY_FILE = "fdsn_station.xml"
OUTPUT_DIR = Path("sena-mseed-3d_plots_100Hz")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PARALLEL = False   
N_WORKERS = 8      


# ==========================================================
# CORE CLASS
# ==========================================================
class SpectralCubeBuilder:
    def __init__(
        self,
        mseed_file,
        inventory_file,
        start_date,
        end_date,
        channel="HHZ",
        window_length_sec=600,
        overlap_fraction=0.5,
        freqmin=0.1,
        freqmax=100.0,
        resample_freq=210.0,
        nfreq=128,
    ):
        self.mseed_file = Path(mseed_file)
        self.inventory = read_inventory(inventory_file)

        self.start_date = UTCDateTime(start_date)
        self.end_date = UTCDateTime(end_date)

        self.channel = channel
        self.window_length_sec = window_length_sec
        self.overlap_fraction = overlap_fraction

        self.freqmin = freqmin
        self.freqmax = freqmax
        self.resample_freq = resample_freq

        self.NDAYS = int((self.end_date.date - self.start_date.date).days) + 1
        self.NHOURS = 24
        self.NFREQ = nfreq

        self.spectra = np.zeros((self.NDAYS, self.NHOURS, self.NFREQ), dtype=np.float32)
        self.counts = np.zeros((self.NDAYS, self.NHOURS), dtype=np.float32)

        self.freqs = None

    # ==========================================================
    # Preprocessing
    # ==========================================================
    def preprocess_trace(self, tr):
        tr.detrend("linear")
        tr.detrend("demean")
        tr.resample(self.resample_freq)
        tr.data = tr.data.astype(np.float32)
        tr.filter("bandpass", freqmin=self.freqmin, freqmax=self.freqmax)
        return tr

    # ==========================================================
    # STFT
    # ==========================================================
    def compute_stft(self, data, fs):
        nperseg = int(self.window_length_sec * fs)
        step = int(nperseg * (1 - self.overlap_fraction))

        if len(data) < nperseg:
            return None, None, None

        window = np.hanning(nperseg)

        spectrogram = []
        times = []

        for start in range(0, len(data) - nperseg, step):
            segment = data[start:start + nperseg]
            segment = segment - np.mean(segment)
            segment *= window

            fft_vals = np.fft.rfft(segment)
            power = np.abs(fft_vals) ** 2

            spectrogram.append(power)
            times.append((start + nperseg // 2) / fs)

        spectrogram = np.array(spectrogram).T
        freqs = np.fft.rfftfreq(nperseg, d=1/fs)
        times = np.array(times) / 3600.0

        return spectrogram, freqs, times

    # ==========================================================
    # Accumulate
    # ==========================================================
    def accumulate_day(self, day_index, spectrogram, freqs, times):
        if self.freqs is None:
            self.freqs = freqs[:self.NFREQ]

        for i, t in enumerate(times):
            hour = int(t)
            if 0 <= hour < 24:
                self.spectra[day_index, hour] += spectrogram[:, i][:self.NFREQ]
                self.counts[day_index, hour] += 1

    # ==========================================================
    # Build cube
    # ==========================================================
    def build_cube(self):
        current_day = self.start_date

        while current_day < self.end_date:
            next_day = current_day + 86400
            day_index = int((current_day - self.start_date) / 86400)

            print(f"Processing day: {current_day.date}")

            try:
                st = read(
                    str(self.mseed_file),
                    starttime=current_day,
                    endtime=next_day
                )
            except Exception:
                current_day = next_day
                continue

            if len(st) == 0:
                current_day = next_day
                continue

            st = st.select(channel=self.channel)
            if len(st) == 0:
                current_day = next_day
                continue

            tr = self.preprocess_trace(st[0].copy())

            result = self.compute_stft(tr.data, tr.stats.sampling_rate)
            if result[0] is None:
                current_day = next_day
                continue

            spectrogram, freqs, times = result
            self.accumulate_day(day_index, spectrogram, freqs, times)

            del st, tr, spectrogram
            gc.collect()

            current_day = next_day

        self.normalize()

    # ==========================================================
    # Normalize
    # ==========================================================
    def normalize(self):
        self.counts[self.counts == 0] = 1
        self.spectra /= self.counts[:, :, None]

    # ==========================================================
    # Outputs
    # ==========================================================
    def get_band_average(self, fmin, fmax):
        band = (self.freqs >= fmin) & (self.freqs <= fmax)
        return self.spectra[:, :, band].mean(axis=2)

    def plot_3d(self, Z, title, output_path, dpi=300):
        days = np.arange(1, self.NDAYS + 1)
        hours = np.arange(24)
        D, H = np.meshgrid(days, hours)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(D, H, Z.T, cmap="viridis", edgecolor="none")

        ax.set_xlabel("Day")
        ax.set_ylabel("Hour")
        ax.set_zlabel("Power")
        ax.set_title(title)

        fig.colorbar(surf, shrink=0.6, label="Power")

        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path}")


# ==========================================================
# FILE PROCESSOR
# ==========================================================
def process_file(mseed_file):
    print(f"\nProcessing: {mseed_file.name}")

    match = re.search(r"(\d{14})_(\d{14})", mseed_file.name)
    if not match:
        print("Skipping (invalid filename)")
        return

    start_date = UTCDateTime(match.group(1))
    end_date = UTCDateTime(match.group(2))

    builder = SpectralCubeBuilder(
        mseed_file=mseed_file,
        inventory_file=INVENTORY_FILE,
        start_date=start_date,
        end_date=end_date,
    )

    builder.build_cube()
    Z = builder.get_band_average(0.1, 0.5)

    output_file = OUTPUT_DIR / f"{mseed_file.stem}_cube.pdf"

    builder.plot_3d(
        Z,
        title=f"{start_date.date} → {end_date.date}",
        output_path=output_file
    )

    del builder, Z
    gc.collect()



if __name__ == "__main__":
    def main():
        files = sorted(BASE_DIR.glob("*.mseed"))

        if PARALLEL:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                executor.map(process_file, files)
        else:
            for f in files:
                process_file(f)

    main()