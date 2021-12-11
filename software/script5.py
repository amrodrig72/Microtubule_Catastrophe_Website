#Import packages
import pandas as pd; import numpy as np; import bebi103

import math; import scipy; import scipy.stats as st; import numba; import tqdm; import warnings

import iqplot; import bokeh.io; import bokeh.plotting; import colorcet; import holoviews as hv

bokeh.io.output_notebook()
hv.extension("bokeh") 

#import data
data3 = pd.read_csv('../data/gardner_mt_catastrophe_only_tubulin.csv', comment= "#")

df = data3[['7 uM', '9 uM', '10 uM', '12 uM', '14 uM']]
#How to rename columns so that they start at 0 and end at 4 for looping
df = df.rename(columns={"7 uM": 0, "9 uM": 1, "10 uM": 2, "12 uM": 3, "14 uM": 4, })
df = pd.melt(df, value_name='Time[s]')
df.rename(columns = {'variable':'Concentration'}, inplace = True)
df = df.dropna()
df = df.reset_index(drop = True)

def log_like_gamma(params, t):
    """Log likelihood for a Gamma distribution."""
    alpha, beta = params
    
    if alpha <= 0 or beta <= 0:
        return -np.inf
    
    return st.gamma.logpdf(t, alpha, loc=0, scale=1/beta).sum()

def gamma_mle(t):
    # Initial guess
    t_bar = np.mean(t)
    beta_guess = t_bar / np.var(t)
    alpha_guess = t_bar * beta_guess

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # We need to crank up the tolerance for this one
        res = scipy.optimize.minimize(
            lambda params, t: -log_like_gamma(params, t),
            (alpha_guess, beta_guess),
            args=(t,),
            method="powell",
            tol=1e-7,
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message) 
        
#Bootstrapping functions 

rg = np.random.default_rng(10000)

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))

def draw_bs_reps_mle(mle_fun, data, args=(), size=1000, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data), *args) for _ in iterator])



results = np.empty((len(df["Concentration"].unique()), 2))

for Concentration, g in df.groupby("Concentration"):
    results[Concentration] = gamma_mle(g["Time[s]"].values)

df_mle = pd.DataFrame(
    data=results,
    columns=["Alpha", "Beta"],
)

colors = colorcet.b_glasbey_category10

reps = {}
for Concentration, g in tqdm.tqdm(df.groupby("Concentration")):
    # Extract time points and mean fluorescence measurements
    t = g["Time[s]"].values

    # Generate bootstrap replicates
    reps[Concentration] = draw_bs_reps_mle(gamma_mle, t, size=5000, progress_bar = False,)
    
p = bokeh.plotting.figure(
    x_axis_label="Alpha",
    y_axis_label="Beta",
    frame_height=400,
    frame_width=400,
)

for Concentration, bs_reps in reps.items():
    # Extract contour lines in Alpha-Beta plane.
    x_line, y_line = bebi103.viz.contour_lines_from_samples(
        x=bs_reps[:, -2], y=bs_reps[:, -2], levels=[0.95]
    )
    
    # Plot the contour lines with fill
    for x, y in zip(x_line, y_line):
        p.line(x, y, line_width=2, color=colors[Concentration], legend_label=f'Concentration {Concentration}')
        p.patch(x, y, fill_color=colors[Concentration], alpha=0.3)
        
p.legend.location = "top_left"

bokeh.io.show(p)