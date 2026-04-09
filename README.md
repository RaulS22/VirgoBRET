### About This Work ###

This work has been done by using open data from INGV.
The aim is to process seismic data from SoS Enatos region.

# Old folder #

First, a study about how to work on seismic data was done by 
using one of the seismic bulletins from INGV. At this step, 
.qml files were used and we could undestand what instances are
necessary to perform a useful workflow (specially when it comes
to spectral analysis). After this, we have tried to obtain data
from the Netherlands station (available at knmi.nl) in order to
have other region to compare, but all of this was without suscess.
Another files downloaded at this preliminar studies of seismic 
data was the Einstein Telescope data (also from INGV). This marked
an important step, as it was at this point that we began working 
with .mseed files and established a standardized workflow for 
seismic data processing.

All of this work is at the /old folder and has been archived just
for a report of what was previously done.

# Actual workflow #

The data from the SoS Enatos region was downloaded month by month 
strating at 00:00:00 of the 1st say, and ending at 23:59:59 pof the
last day of the respective month. To keep an standart of the data,
it started at 21/06/2021 and ended at 31/12/2025 (DD/MM/YYYY). Since
it starts at the very begining of the measurements and ends at the
last complete year at the beging of this work.

A preliminar study of complete month was done first in order to debugg
and to optimize the codes. For this, the folder "/SENA-mseed" was used,
containing a folder of the complete month of jan/2024 (explained at the 
info.txt file). 
The files at "/sena-jan24-raw-plots" folders were the first ploted data
of amplitude versus time. At "/sena-jan24-filtered-test" we've made a
cutting on microseismic data. Finally, at "/sena-jan24-filtered-test"
we've obtained the spectogram of each day.

With all of this in mind, we could perform a PSD analysis of each month
of the year searching for patterns. Some of our results are at the folder
"/sena-mseed-3d_plots".

With an workflow stablished, we have proceded to complete year analysis.
The data is at the /SENA-files and it is shorted by year at the
respective folders for each year. the "/teste" folder contains the
data from june/2021 and has been used for the first tests and debuggs.

# Working on #

Process the velocity of seismic data
Optmizing the analisys of microseismic data