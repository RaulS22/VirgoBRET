# About data used at this work #

This work has been done by using open data from INGV.
At firste, our aim was to process seismic data from SoS Enatos region.
With the sucess of our preliminary analysis, and having stablished a
method, we have extended our analysis to the Virgo Network

Data available at: https://eida.ingv.it/en/getdata/ 
for SoS Enatos region it is necessary to search for MN SENA  and for
Virgo it is necessary to search for VR VRG0x.

After the SoS Enatos region, we have found a better way to obtaining data
using fdsnwsscripts. The file [download_virgo.sh](download_virgo.sh) provides
a way to download Virgo data for all stations and channels during O3b run. The
file [year_virgo.sh](year_virgo.sh) downloads the data for each year, for the 
central station and for the HH3 channel.

We are very thankful for the the [fdsnwsscripts](https://github.com/GEOFN/fdsnws_scripts)
project because we could stablish a more uniform and efficient way to
obtain the data rather than just downloading file by file at the [eida getdata](https://eida.ingv.it/en/getdata/)
website.

### Ongoing Work ###

After performing the q-Transform, the graph was divided into 
30 bins of frequency and 41 bins of time. This matrix was then
flattened into a 1230 vector. These vectors were combined into
a .parquet file (a format similar to .cvs but with some advantages).
With this in hand, dimensionality reduction has been performed. The
first approach was Principal Component Analysis (PCA). Then we 
advanced to t-Distributed Stochastic Neighbor Embedding (t-SNE),
Multidimensional Scaling (MDS) and Uniform Manifold Approximation
and Projection (UMAP). Each one of these methods has its own 
properties and features, and this is what we are putting effort into
understanding right now so we can improve our analysis.

We are also trying to stablish a correlation between classes of seismic
events and glitches using these dimensionality reduction techniques.

At this step, the main results are saved in notebook files in order to
make the presentation of the results simpler to colleagues and everyone
who is interested so the pearson does not have to download GB of data
just to make a plot. Unfortunately, 3d projections and interactive graphs
aren't available since they run locally. We are trying our best to improve this.

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

Update: since the beginning of this work some things have changed. After the Sos
Enatos data was processed by using the code [qT_opt.py](qT_opt.py), we proceeded to
dowload the Virgo data using the scripts .sh. They are very intuitive and the
[fdsnwsscripts readthedocs](https://fdsnwsscripts.readthedocs.io/en/latest/) page
provides a very complete documentation of this tool. 

### Perspectives ###

Study of the periodicity of seismic events.
Classification of seismic events trough band-limiting.
Correlation of classes.