import numpy as np
import matplotlib.pyplot as plt
from obspy import read_events
from obspy.clients.fdsn import Client

# ==========================================================
# Plot Style (LaTeX-quality)
# ==========================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "figure.figsize": (8,5),
})

##########################################################

"""
Some documentation:

https://www.einstein-telescope.it/en/2025/12/02/einstein-telescope-a-new-ingv-center-to-map-the-subsurface-of-sardinia/
With its creation, INGV aims to strengthen its scientific contribution to Italy's candidacy to host the Einstein 
Telescope (ET) gravitational-wave observatory in the area of the former Sos Enattos mine, in the province of Nuoro.

https://www.einstein-telescope.it/en/et-in-italy/#sos-enattos
Finally, in the area around Nuoro, between the municipalities of Bitti, Lula and Onanì, there are large expanses 
of rural areas with very low population density and, therefore, limited anthropogenic and industrial activity.

Data from INGV: http://www.eida.ingv.it/en/ (não adiantou em nada, na verdade)
Notar que não é https 

Network -> Station -> Location -> Channel (SEED Reference Manual FDSN)
Recurso extra: IRIS Station Monitor https://www.iris.edu/app/station_monitor/

3M (2024-2029): Seismic Arrays for the noise characterization of the Sardinia Einstein Telescope candidacy (SANSET)
https://www.fdsn.org/networks/detail/3M_2024/ 

5J (2014-2016): The Sardinia Passive Array Experiment (SPAE)
https://www.fdsn.org/networks/detail/5J_2014/
    - Cimini G. B. (2014). The Sardinia Passive Array Experiment (SPAE) [Data set]. 
      International Federation of Digital Seismograph Networks. https://doi.org/10.7914/SN/5J_2014
Data avliability: http://webservices.ingv.it/fdsnws/dataselect/1/

6P (2019-2024): Einstein Telescope Sardinia Seismic Network
https://www.fdsn.org/networks/detail/6P_2019/
Data for this network is not currently available through FDSN web services.

7P (2021-2022): Einstein Telescope Sardinia Temporary Seismic Arrays
https://www.fdsn.org/networks/detail/7P_2021/
Data avaliability: http://webservices.ingv.it/fdsnws/dataselect/1/
"""


# Open Data: https://data.ingv.it/
# Einstein Telescope: https://data.ingv.it/en/dataset/1089#additional-metadata
# Available at: https://eida.ingv.it/en/network/3M_2024