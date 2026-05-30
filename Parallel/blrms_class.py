import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

#TODO: implement quartis = np.quantile(dados, [0.25, 0.5, 0.75])


class BLRMSProcessor:

    def __init__(self, processed_stream, bands, window=60):
        self.processed_stream = processed_stream
        self.bands = bands
        self.window = window
        self.results = {}

    # ======================================================
    # RMS
    # ======================================================

    @staticmethod
    def rms(x):
        return np.sqrt(
            np.mean(x**2)
        )

    # ======================================================
    # COMPUTE BLRMS
    # ======================================================

    def compute_blrms(self):
        for band_name, (fmin, fmax) in self.bands.items():
            print(f"\nProcessing band: {band_name}")
            daily_values = []
            daily_times = []
            for tr in self.processed_stream:
                tr_filt = tr.copy()

                tr_filt.filter(
                    "bandpass",
                    freqmin=fmin,
                    freqmax=fmax,
                    corners=4,
                    zerophase=True
                )

                fs = tr_filt.stats.sampling_rate
                data = tr_filt.data

                npts_window = int(self.window * fs)
                rms_values = []

                for i in range(
                    0,
                    len(data) - npts_window,
                    npts_window
                ):

                    chunk = data[
                        i:i + npts_window
                    ]

                    rms_values.append(self.rms(chunk))

                if len(rms_values) == 0:
                    continue

                daily_median = np.median(rms_values)
                daily_values.append(daily_median * 1e9)  # m/s -> nm/s

                daily_times.append(tr.stats.starttime.datetime)
            self.results[band_name] = (daily_times, daily_values)

    # ======================================================
    # PLOT
    # ======================================================

    def plot(self):
        plt.figure(figsize=(12, 5))

        for band_name, (times, values) in self.results.items():

            plt.plot(times, values, marker="o", label=band_name)

            plt.xlabel("Date")
            plt.ylabel("Daily Median Ground Velocity [nm/s]")
            plt.yscale("log")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("blrms_plot.pdf", dpi=300)