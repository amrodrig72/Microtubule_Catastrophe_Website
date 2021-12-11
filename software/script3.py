#Import packages
import pandas as pd; import numpy as np; import bebi103

import math; import scipy; import scipy.stats as st; import numba; import tqdm; import warnings

import iqplot; import bokeh.io; import bokeh.plotting; import colorcet; import holoviews as hv

bokeh.io.output_notebook()
hv.extension("bokeh")  

#import data
data = pd.read_csv('../data/gardner_time_to_catastrophe_dic_tidy.csv', header=[0])

labeled = data.loc[data["labeled"] == True, "time to catastrophe (s)"].values
unlabeled = data.loc[data["labeled"] == False, "time to catastrophe (s)"].values


p = iqplot.ecdf(
    data=data, cats="labeled", q="time to catastrophe (s)", conf_int=True
)

p.legend.title = "labeled"