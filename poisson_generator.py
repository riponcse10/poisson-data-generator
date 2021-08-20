#import numpy as np
#import matplotlib.pyplot as plt

#s = np.random.poisson(5, 1000)
#count, bins, ignored = plt.hist(s, density=True)
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
import seaborn as sns

from pandas import DataFrame

import plotly.express as px


# initializing random values
data = np.random.poisson(60, 1000)
#data = np.random.poisson(lam=(5., 1., 60.), size=(1000, 3))
  
df = DataFrame (data)

#df = px.data.tips()
fig = px.histogram(data, nbins=15)
fig.update_layout(bargap=0.05)
fig.update_traces(histnorm="probability", selector=dict(type='histogram'))
fig.update_layout(
    title="Plot Title",
    xaxis_title="Inter arrival time",
    yaxis_title="Probability",
    legend_title="",
    font=dict(
        size=38,
        color="Black"
    )
)
fig.show()


