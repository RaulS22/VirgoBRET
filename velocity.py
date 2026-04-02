import numpy as np
import matplotlib.pyplot as plt
from obspy import read, read_inventory
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import gc
