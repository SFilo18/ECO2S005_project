"""
Econometric analysis wrappers for the Bitcoin fees project.

Thin layer over statsmodels and scipy that produces tidy outputs
suited to the report's regression tables and diagnostic discussions.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller, kpss, coint, grangercausalitytests
from statsmodels.iolib.summary2 import summary_col
from scipy import stats


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def summary_stats(df: pd.DataFrame, cols: Optional[list] = None) -> pd.DataFrame:
    """
    Descriptive statistics for numeric columns.

    Beyond pandas .describe(): also reports skewness and kurtosis,
    which matter a lot for crypto data (heavy tails, right skew).

    Returns one row per variable with columns:
    N, mean, median, std, min, max, skew, kurtosis.
    """
    # TODO
    pass


def correlation_matrix(df: pd.DataFrame, cols: list,
                        method: str = 'pearson',
                        plot: bool = False) -> pd.DataFrame:
    """Pearson (default) or Spearman correlation matrix; optionally plot a heatmap."""
    # TODO
    pass


# ============================================================================
# STATIONARITY
# ============================================================================

def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test.

    H0: series has a unit root (non-stationary).
    p < 0.05 → reject H0 → series is stationary.
    """
    # TODO: stat, p, lags, nobs, crit, _ = adfuller(series.dropna())
    # return dict.
    pass


def kpss_test(series: pd.Series, regression: str = 'c') -> dict:
    """
    KPSS test. Opposite null hypothesis to ADF.

    H0: series is stationary.
    p < 0.05 → reject H0 → series is non-stationary.

    Running ADF and KPSS together gives a more robust conclusion.
    """
    # TODO
    pass


def stationarity_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Run ADF and KPSS on each column; return one-row-per-variable summary.

    Columns: variable, adf_stat, adf_p, kpss_stat, kpss_p, verdict.
    """
    # TODO: 'verdict' could be 'stationary', 'non-stationary', or 'inconclusive'
    # based on whether ADF and KPSS agree.
    pass


# ============================================================================
# REGRESSION
# ============================================================================

def run_ols(formula: str, df: pd.DataFrame,
            robust: bool = True, maxlags: int = 7):
    """
    Fit an OLS model with optional HAC (Newey-West) robust standard errors.

    Use robust=True for time-series data — it's the right default here
    because residuals will be autocorrelated and heteroskedastic.

    Returns a fitted statsmodels RegressionResultsWrapper.
    Use .summary() to print, or .params / .pvalues / .rsquared to extract.
    """
    if robust:
        return smf.ols(formula, data=df).fit(
            cov_type='HAC', cov_kwds={'maxlags': maxlags}
        )
    return smf.ols(formula, data=df).fit()


def compare_models(models: list, names: Optional[list] = None):
    """
    Side-by-side regression table using statsmodels.summary_col.

    Produces the headline "as we add congestion variables, the price
    coefficient shrinks" table for the report.
    """
    return summary_col(
        models,
        stars=True,
        model_names=names,
        info_dict={
            'R²':      lambda x: f'{x.rsquared:.3f}',
            'Adj. R²': lambda x: f'{x.rsquared_adj:.3f}',
            'N':       lambda x: f'{int(x.nobs)}',
        },
    )


# ============================================================================
# STRUCTURAL BREAKS
# ============================================================================

def chow_test(df: pd.DataFrame, breakpoint: str, formula: str) -> dict:
    """
    Chow test for a structural break at `breakpoint` (a date string).

    H0: coefficients are identical before and after the breakpoint.
    p < 0.05 → reject H0 → break exists, regimes are different.

    Returns dict: f_statistic, p_value, df_num, df_denom, n_before, n_after.
    """
    # Implementation outline:
    #   before = df[df.index <  breakpoint]
    #   after  = df[df.index >= breakpoint]
    #   rss_p  = smf.ols(formula, df).fit().ssr
    #   rss_b  = smf.ols(formula, before).fit().ssr
    #   rss_a  = smf.ols(formula, after).fit().ssr
    #   k = len(smf.ols(formula, df).fit().params)
    #   n = len(df)
    #   f  = ((rss_p - (rss_b + rss_a)) / k) / ((rss_b + rss_a) / (n - 2*k))
    #   p  = 1 - stats.f.cdf(f, k, n - 2*k)
    pass


def chow_test_multiple(df: pd.DataFrame, breakpoints: list,
                        formula: str) -> pd.DataFrame:
    """Run chow_test at each breakpoint; return a tidy DataFrame of results."""
    # TODO
    pass


# ============================================================================
# OTHER TIME-SERIES TESTS
# ============================================================================

def granger_test(df: pd.DataFrame, cause: str, effect: str,
                  maxlag: int = 10) -> pd.DataFrame:
    """
    Does `cause` Granger-cause `effect`?

    Runs grangercausalitytests for lags 1..maxlag and returns
    a clean DataFrame of p-values (one column per lag, one test per row).
    """
    # TODO: wrap grangercausalitytests, extract the 'ssr_ftest' p-values.
    pass


def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """
    Engle-Granger cointegration test.

    H0: no cointegration. p < 0.05 → variables share a long-run
    equilibrium relationship even if both are non-stationary.

    Returns dict: statistic, p_value, critical_values.
    """
    # TODO: stat, p, crit = coint(y, x); return dict.
    pass


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def plot_series(df: pd.DataFrame, cols: list, log: bool = False,
                title: Optional[str] = None):
    """Quick time-series plot of one or more columns, with optional log scale."""
    # TODO
    pass


def plot_residuals(model, lags: int = 40):
    """
    Diagnostic 2x2 panel for regression residuals:
    time series, histogram + KDE, Q-Q plot, ACF.

    Use this after every regression to check assumptions visually.
    """
    # TODO
    pass