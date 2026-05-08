import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
from obspy.clients.fdsn import Client
from obspy import read

import parallel_class as pc
import blrms_class as blrms

processor = pc.SeismicProcessor(mseed_file="SENA-files/2021/eida_response_MN-SENA_20211201000000_20211231235959.mseed",
        chunk_duration=6 * 3600, n_processes=8)

processed_stream = processor.run()
print("\nProcessing complete!")

bands = {
    "microseism": (0.1, 1),
    "anthropogenic": (1, 10),
    "high_freq": (10, 20)
}

blrms = blrms.BLRMSProcessor(processed_stream, bands=bands, window=60)

blrms.compute_blrms()
blrms.plot()