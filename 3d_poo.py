import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_inventory, UTCDateTime
from pathlib import Path
import gc

input_dir = Path("/home/rauls/Desktop/GithubITA/VirgoBRET/SENA-mseed")

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
        freqmax=1.0,
        resample_freq=2.1,
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

        self.NDAYS = int((self.end_date - self.start_date) / 86400)
        self.NHOURS = 24
        self.NFREQ = nfreq

        self.spectra = np.zeros((self.NDAYS, self.NHOURS, self.NFREQ))
        self.counts = np.zeros((self.NDAYS, self.NHOURS))

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

    # FFT
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
        times = np.array(times) / 3600.0  # hours

        return spectrogram, freqs, times

    # Cube
    def accumulate_day(self, day_index, spectrogram, freqs, times):
        if self.freqs is None:
            self.freqs = freqs[:self.NFREQ]

        for i, t in enumerate(times):
            hour = int(t)

            if hour < 0 or hour >= 24:
                continue

            spectrum_slice = spectrogram[:, i][:self.NFREQ]

            self.spectra[day_index, hour] += spectrum_slice
            self.counts[day_index, hour] += 1

    # Loop
    def build_cube(self):
        current_day = self.start_date

        while current_day < self.end_date:
            next_day = current_day + 86400
            day_index = int((current_day - self.start_date) / 86400)

            print(f"Processing day: {current_day.date}")

            st = read(
                str(self.mseed_file),
                starttime=current_day,
                endtime=next_day
            )

            if len(st) == 0:
                current_day = next_day
                continue

            st = st.select(channel=self.channel)

            if len(st) == 0:
                current_day = next_day
                continue

            tr = st[0].copy()
            tr = self.preprocess_trace(tr)

            data = tr.data
            fs = tr.stats.sampling_rate

            result = self.compute_stft(data, fs)

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
    # Normalize cube
    # ==========================================================
    def normalize(self):
        self.counts[self.counts == 0] = 1
        self.spectra /= self.counts[:, :, None]
    '''
    It is important to note that the normalization step is crucial for ensuring that the spectral power
    values in the cube are comparable across different days and hours, especially since some time bins
    may have more data than others. By dividing the accumulated spectra by the count of segments, we can 
    mitigate the effects of varying data availability and obtain a more accurate representation of the 
    underlying seismic activity.

    https://dsp.stackexchange.com/questions/19311/stft-why-overlapping-the-window
    '''

    # ==========================================================
    # Extract slice
    # ==========================================================
    def get_frequency_slice(self, freq_target=None, freq_bin=None):
        if freq_bin is None:
            freq_bin = np.argmin(np.abs(self.freqs - freq_target))

        Z = self.spectra[:, :, freq_bin]
        return Z

    def get_band_average(self, fmin, fmax):
        band = (self.freqs >= fmin) & (self.freqs <= fmax)
        return self.spectra[:, :, band].mean(axis=2)

    def plot_3d(
        self,
        Z,
        title="3D Spectral Cube",
        output_path=None,
        dpi=300,
):
        days = np.arange(1, self.NDAYS + 1)
        hours = np.arange(self.NHOURS)

        D, H = np.meshgrid(days, hours)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            D,
            H,
            Z.T,
            cmap="viridis",
            edgecolor="none"
        )

        ax.set_xlabel("Day")
        ax.set_ylabel("Hour")
        ax.set_zlabel("Power")
        ax.set_title(title)

        fig.colorbar(surf, shrink=0.6, label="Power")

        # ==========================
        # Save or show
        # ==========================
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(
                output_path,
                format="pdf",
                dpi=dpi,
                bbox_inches="tight"
            )

            print(f"Saved 3D plot to: {output_path}")
            plt.close(fig)
        else:
            plt.show()




if __name__ == "__main__":
    builder = SpectralCubeBuilder(
    mseed_file=input_dir/"eida_response_MN-SENA_20210801000000_20210831235959.mseed",
    inventory_file="fdsn_station.xml",
    start_date="2021-08-01",
    end_date="2021-08-31",
    )

    builder.build_cube()
    Z = builder.get_band_average(0.1, 0.5)

    builder.plot_3d(
    Z,
    title="August Spectral Cube",
    output_path="sena-mseed-3d_plots/august_cube.pdf")