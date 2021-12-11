#Import packages
import pandas as pd; import numpy as np; import bebi103

import math; import scipy; import scipy.stats as st; import numba; import tqdm; import warnings

import iqplot; import bokeh.io; import bokeh.plotting; import colorcet; import holoviews as hv

bokeh.io.output_notebook()
hv.extension("bokeh") 

#import data
data3 = pd.read_csv('../data/gardner_mt_catastrophe_only_tubulin.csv', comment= "#")

# First we need to tidy our data
df = data3[['7 uM', '9 uM', '10 uM', '12 uM', '14 uM']]
df = df.rename(columns={"7 uM": 7, "9 uM": 9, "10 uM": 10, "12 uM": 12, "14 uM": 14, })
df = pd.melt(df, value_name='Time[s]')
df.rename(columns = {'variable':'Concentration'}, inplace = True)
df = df.dropna()
df = df.reset_index(drop = True)

#Isolate our 12uM concntration data 
df_12uM = df.loc[df['Concentration'] == 12]
t12 = df_12uM['Time[s]'].values

#Now lets define some functions to get our MLE parameter estimates for gamma distriubtion  
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

# Okay now lets find our mle for 12
g_alpha_mle_12, g_beta_mle_12 =gamma_mle(t12)

#Now lets define some functions to get our MLE parameter estimates for gamma distriubtion 
def log_like(beta1, t):
    """Compute the log likelihood for a given value of β1,
    assuming Δβ is set so that the dervitive of the log
    likelihood with respect to β1 vanishes."""
    n = len(t)
    tbar = np.mean(t)
    beta1_tbar = beta1 * tbar
    
    if beta1_tbar > 2 or beta1_tbar < 1:
        return np.nan

    if np.isclose(beta1_tbar, 2):
        return -2 * n * (1 + np.log(tbar) - np.log(2)) + np.sum(np.log(t))
        
    if np.isclose(beta1_tbar, 1):
        return -n * (1 + np.log(tbar))
            
    delta_beta = beta1 * (2 - beta1 * tbar) / (beta1 * tbar - 1)

    ell = n * (np.log(beta1) + np.log(beta1 + delta_beta) - np.log(delta_beta))
    ell -= n * beta1_tbar
    ell += np.sum(np.log(1 - np.exp(-delta_beta * t)))
    
    return ell

def dlog_like_dbeta1(beta1, t):
    """Returns the derivative of the log likelihood w.r.t. Δβ
    as a function of β1, assuming Δβ is set so that the dervitive 
    of the log likelihood with respect to β1 vanishes."""
    n = len(t)
    tbar = np.mean(t)
    beta1_tbar = beta1 * tbar
    
    if beta1_tbar > 2 or beta1_tbar < 1:
        return np.nan

    if np.isclose(beta1_tbar, 2) or np.isclose(beta1_tbar, 1):
        return 0.0
            
    delta_beta = beta1 * (2 - beta1 * tbar) / (beta1 * tbar - 1)
    
    exp_val = np.exp(-delta_beta * t)
    sum_term = np.sum(t * exp_val / (1 - exp_val))
    
    return -n / delta_beta + n / (beta1 + delta_beta) + sum_term


def mle_two_step(t, nbeta1=500):
    """Compute the MLE for the two-step model."""
    # Compute ∂ℓ/∂Δβ for values of beta_1
    tbar = np.mean(t)
    beta1 = np.linspace(1 / tbar, 2 / tbar, nbeta1)
    deriv = np.array([dlog_like_dbeta1(b1, t) for b1 in beta1])
    
    # Add the roots at the edges of the domain
    beta1_vals = [1 / tbar, 2 / tbar]
    ell_vals = [log_like(beta1_vals[0], t), log_like(beta1_vals[1], t)]
    
    # Find all sign flips between the edges of the domain
    sign = np.sign(deriv[1:-1])
    inds = np.where(np.diff(sign))[0]
    
    # Perform root finding at the sign flips
    for i in inds:
        b1 = scipy.optimize.brentq(dlog_like_dbeta1, beta1[i+1], beta1[i+2], args=(t,))
        beta1_vals.append(b1)
        ell_vals.append(log_like(b1, t))
        
    # Find the value of beta1 that gives the maximal log likelihood
    i = np.argmax(ell_vals)
    beta1 = beta1_vals[i]

    # Compute beta 2
    if np.isclose(beta1, 1 / tbar):
        delta_beta = np.inf
    else:
        delta_beta = beta1 * (2 - beta1 * tbar) / (beta1 * tbar - 1)

    beta2 = beta1 + delta_beta
    
    return np.array([beta1, beta2])

# And now return Beta1/2 for our two-step model
beta1, beta2 = mle_two_step(t12)

#Now we can assess our models. 

#Model assessment gamma 
rg = np.random.default_rng()

single_gamma = np.array(
    [rg.gamma(g_alpha_mle_12, 1 / (g_beta_mle_12), size=len(t12)) for _ in range(100000)]
)

p1 = bebi103.viz.predictive_ecdf(
    samples=single_gamma, data=t12, discrete=True, x_axis_label="Gamma 12[uM] - Time[s]"
)

p2 = bebi103.viz.predictive_ecdf(
    samples=single_gamma, data=t12, diff='ecdf', discrete=True, x_axis_label="Gamma 12[uM] - Time[s]"
)

#Model assessment two-step
rg = np.random.default_rng()

single_two_step_b_is_b = np.array(
    [rg.gamma(2, 1 / (beta1), size=len(t12)) for _ in range(100000)]
)

p3 = bebi103.viz.predictive_ecdf(
    samples=single_two_step_b_is_b, data=t12, discrete=True, x_axis_label="Two Step 12[uM] - Time[s]"
)

p4 = bebi103.viz.predictive_ecdf(
    samples=single_two_step_b_is_b, data=t12, diff='ecdf', discrete=True, x_axis_label="Two Step 12[uM] - Time[s]"
)