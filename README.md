# About data used at this work #

This work has been done by using open data from INGV.
The aim is to process seismic data from SoS Enatos region.

Data available at: https://eida.ingv.it/en/getdata/ 
for SoS Enatos region it is necessary to search for SENA.

### Ongoing Work ###

What has been done now is the q-Transform of data. 

The approach to stablish a "SNR" of the signal is the 
ratio between a STA (short time average) and LTA (long
time average) and had promissing results unstil now. 
With this in hand, a "cutoff" is set and we are able to
get the times of the events that we want. 

After this, the q-Transform is created, using GWPy. 
The optimal value of the parameters are still being 
tested before we proceed with the dimensional reduction
techniques that we want to use so we can have coherent 
data.

### Other Results ###

A study of microseisms was perfomed by making use of BLRMS
(Band-Limited Root Mean Square). With this, it was possible
to observe that diferent bands will have diferent responses.

CEEMDAN (Complete Ensemble Empirical Mode Decomposition with
Adaptive Noise) was used in a file in order as a commissioning
and benchmark of this type of algorithm for future pespectives.

### Old folder ###

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

### About the structure of this work ###

The data from the SoS Enatos region was downloaded month by month 
strating at 00:00:00 of the 1st say, and ending at 23:59:59 pof the
last day of the respective month. To keep an standart of the data,
it started at 21/06/2021 and ended at 31/12/2025 (DD/MM/YYYY). Since
it starts at the very begining of the measurements and ends at the
last complete year at the beging of this work.

A preliminar study of complete month was done first in order to debug
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
data from june/2021 and has been used for the first tests and debugs.

Note from May: now these folders are located at /FFT_PSD_Tests.

We have obtained a way to develop the calculations of the FFT and PSD
of the data. An atempt of paralelizing was done, with openmp, with promissing
results. These results are stored at the /FFT_PSD_Tests folder.
Some of the paralelizing steps are at the folder /Parallel folder.

We also processed the velocity of seismic data in order to obtain a
physical meaning and to have a better understanding of data so the classification
could be done in a better way.
We had some issues with the removing of the station response, so there is a little
debug program at the /Station folder, so this specific test can be done quickly.

Optmizing the computaional time of the analisys of
our microseismic data has been an important step, since we want to implement
the q-transform and use dimensional reduction techniques (such as t-SNE).

### Perspectives ###

Study of the periodicity of seismic events.

Classification of seismic events trough band-limiting.
