import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def adf_test(ts,verbose=True):
    dftest = adfuller(ts, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
        
    if verbose:
        print("Results of Dickey-Fuller Test:")
        print(dfoutput)
    
    return dftest[1]
    
def kpss_test(ts, verbose=True):
    kpsstest = kpss(ts, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
        
    if verbose:
        print("Results of KPSS Test:")
        print(kpss_output)
    
    return kpsstest[1]
    
def ts_plots(ts, verbose=False):
    f, ax = plt.subplots(2,1, figsize=(10,5))
    
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    
    adf_pvalue = adf_test(ts, verbose=verbose)
    kpss_pvalue = kpss_test(ts, verbose=verbose)
    ts.plot.hist(title=f'Time Series Plots\nADF Test:{adf_pvalue}\nKPSS Test:{kpss_pvalue}', ax=ax[0])
    ts.plot(ax=ax[1])
    
    plt.tight_layout()
    plt.show();

def acf_pacf_plots(y, lags=20, figsize=(12,4)):
    f, ax = plt.subplots(1,2, figsize=figsize)
    plot_acf(y, lags=lags, ax=ax[0])
    plot_pacf(y, lags=lags, ax=ax[1], method='ywmle')
    plt.tight_layout()

def ljung_box(ts, lags=40, dof=0):
    res = acorr_ljungbox(ts, lags=lags, return_df=True, model_df=dof)
    return not res[res.lb_pvalue <= 0.05].empty

def ts_res_analysis(valid_res, name, lags, dof):
    stationary_dict = {}
    acf_dict = {}
    stationary_arr = []
    acf_arr = []

    for i in range(50):
        stationary_dict[i+1] = 0
        acf_dict[i+1] = 0

    for i in range(50):
        for s in range(10):
            ts = valid_res.loc[(valid_res.item==i+1) & (valid_res.store==s+1)][['sales', name]].copy()
            resid = ts.sales - ts[name]
            adf_pvalue = adf_test(resid, verbose=False)
            kpss_pvalue = kpss_test(resid, verbose=False)
            if (adf_pvalue > 0.05) and (kpss_pvalue > 0.05):
                stationary_dict[i+1] += 1
                stationary_arr.append([i+1, s+1])
            if ljung_box(resid, lags=lags, dof=dof):
                acf_dict[i+1] += 1
                acf_arr.append([i+1, s+1])
    
    return stationary_dict, acf_dict, stationary_arr, acf_arr

def plot_residuals(valid_res, s, i, name, lags):
    ts = valid_res.loc[(valid_res.store==s) & (valid_res.item==i)][['sales', name]].copy()
    resid = ts.sales - ts[name]
    ts_plots(resid)
    acf_pacf_plots(resid, lags=lags)


