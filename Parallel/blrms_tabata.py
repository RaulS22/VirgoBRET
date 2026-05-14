import numpy as np
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt
from pathlib import Path


class BLRMS:
    """ Band-Limited Root Mean Square calculator. """

    def __init__(self, filepath, window_size, freqmin, freqmax):
        """
        Parameters
        ----------
        filepath : str
            Path to .mseed file

        window_size : float
            Window size in seconds

        freqmin : float
            Lower bandpass frequency

        freqmax : float
            Upper bandpass frequency
        """

        self.filepath = filepath
        self.window_size = window_size
        self.freqmin = freqmin
        self.freqmax = freqmax

        self.trace = None
        self.filtered_data = None
        self.window_rms = None

    def load_data(self):
        """
        Load mseed file.
        """

        st = read(self.filepath)
        self.trace = st[0]

    def bandpass_filter(self):
        """
        Apply bandpass filter.
        """

        tr = self.trace.copy()

        tr.filter(
            "bandpass",
            freqmin=self.freqmin,
            freqmax=self.freqmax,
            corners=4,
            zerophase=True
        )

        self.filtered_data = tr.data.astype(np.float32) #faster than float64, result of some benchmarks i've made

    def remove_outliers_iqr(self):
        q1 = np.percentile(self.filtered_data, 25)
        q3 = np.percentile(self.filtered_data, 75)

        mask = ((self.filtered_data >= q1) & (self.filtered_data <= q3)) #interquartile range
        self.filtered_data = self.filtered_data[mask]

    def compute_blrms(self):
        """
        Compute RMS value for each window.
        """

        fs = self.trace.stats.sampling_rate
        samples_per_window = int(self.window_size * fs)
        n_windows = len(self.filtered_data) // samples_per_window
        rms_values = []

        for i in range(n_windows):
            start = i * samples_per_window
            end = start + samples_per_window
            window = self.filtered_data[start:end]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)

        self.window_rms = np.array(rms_values)

    def median_blrms(self):
        """
        Median RMS value across all windows.
        """

        return np.median(self.window_rms)

    def run(self):
        self.load_data()
        self.bandpass_filter()
        self.remove_outliers_iqr()
        self.compute_blrms()
        return self.window_rms
    
if __name__ == "__main__":
    file_path = "Parallel/Processed/012023processed.mseed"

    blrms = BLRMS(
    filepath=file_path,
    window_size=60,   # seconds
    freqmin=0.1,
    freqmax=3.0
    )

    rms_per_window = blrms.run()
    median_value = blrms.median_blrms()

    print(rms_per_window)
    print("Median BLRMS:", median_value)