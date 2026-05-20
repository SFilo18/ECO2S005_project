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

HALVING_DATES = [
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-19',
]

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
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    rows = {}
    for col in cols:
        s = df[col].dropna()
        rows[col] = {
            'N':        int(s.shape[0]),
            'mean':     s.mean(),
            'median':   s.median(),
            'std':      s.std(),
            'min':      s.min(),
            'max':      s.max(),
            'skew':     stats.skew(s),
            'kurtosis': stats.kurtosis(s),   # excess kurtosis (normal = 0)
        }

    out = pd.DataFrame(rows).T
    # Preserve column order; keep N as a clean integer
    out = out[['N', 'mean', 'median', 'std', 'min', 'max', 'skew', 'kurtosis']]
    out['N'] = out['N'].astype(int)
    return out


def correlation_matrix(df: pd.DataFrame, cols: list,
                        method: str = 'pearson',
                        plot: bool = False) -> pd.DataFrame:
    """Pearson (default) or Spearman correlation matrix; optionally plot a heatmap."""
    corr = df[cols].corr(method=method)

    if plot:
        fig, ax = plt.subplots(figsize=(0.9 * len(cols) + 2, 0.9 * len(cols) + 1))
        sns.heatmap(
            corr,
            annot=True,          # write the correlation value in each cell
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1, vmax=1,     # fix the scale so colors are comparable across plots
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax,
        )
        ax.set_title(f'{method.capitalize()} correlation matrix')
        plt.tight_layout()
        plt.show()

    return corr


# ============================================================================
# STATIONARITY
# ============================================================================

def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test.

    H0: series has a unit root (non-stationary).
    p < 0.05 → reject H0 → series is stationary.
    """
    stat, p, lags, nobs, crit, _ = adfuller(series.dropna())
    return {
        'test':      'ADF',
        'statistic': stat,
        'p_value':   p,
        'lags':      lags,
        'n_obs':     nobs,
        'crit_1%':   crit['1%'],
        'crit_5%':   crit['5%'],
        'crit_10%':  crit['10%'],
        'stationary': p < 0.05,    # convenience boolean for the verdict logic later
    }


def kpss_test(series: pd.Series, regression: str = 'c') -> dict:
    """
    KPSS test. Opposite null hypothesis to ADF.

    H0: series is stationary.
    p < 0.05 → reject H0 → series is non-stationary.

    Running ADF and KPSS together gives a more robust conclusion.
    """
    import warnings
    from statsmodels.tools.sm_exceptions import InterpolationWarning

    with warnings.catch_warnings():
        # KPSS p-values are interpolated from a table and capped at [0.01, 0.10];
        # statsmodels warns when the stat is outside that range. Expected behaviour.
        warnings.simplefilter('ignore', InterpolationWarning)
        stat, p, lags, crit = kpss(series.dropna(), regression=regression, nlags='auto')

    return {
        'test':       'KPSS',
        'statistic':  stat,
        'p_value':    p,
        'lags':       lags,
        'crit_1%':    crit['1%'],
        'crit_5%':    crit['5%'],
        'crit_10%':   crit['10%'],
        'stationary': p >= 0.05,   # NOTE: opposite direction to ADF
    }    


def stationarity_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Run ADF and KPSS on each column; return one-row-per-variable summary.

    Columns: variable, adf_stat, adf_p, kpss_stat, kpss_p, verdict.
    """
    rows = []
    for col in cols:
        adf = adf_test(df[col])
        kpss_ = kpss_test(df[col])

        adf_says_stationary = adf['stationary']    # ADF: p < 0.05
        kpss_says_stationary = kpss_['stationary']  # KPSS: p >= 0.05

        # Both tests agree → confident verdict.
        # They disagree → inconclusive (often trend-stationary or near unit root).
        if adf_says_stationary and kpss_says_stationary:
            verdict = 'stationary'
        elif not adf_says_stationary and not kpss_says_stationary:
            verdict = 'non-stationary'
        else:
            verdict = 'inconclusive'

        rows.append({
            'variable':  col,
            'adf_stat':  adf['statistic'],
            'adf_p':     adf['p_value'],
            'kpss_stat': kpss_['statistic'],
            'kpss_p':    kpss_['p_value'],
            'verdict':   verdict,
        })

    return pd.DataFrame(rows).set_index('variable')

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
    bp = pd.Timestamp(breakpoint)
    before = df[df.index < bp]
    after  = df[df.index >= bp]

    # Fit pooled and the two sub-period regressions
    m_pooled = smf.ols(formula, data=df).fit()
    m_before = smf.ols(formula, data=before).fit()
    m_after  = smf.ols(formula, data=after).fit()

    rss_p = m_pooled.ssr
    rss_b = m_before.ssr
    rss_a = m_after.ssr

    k = len(m_pooled.params)              # number of parameters (incl. intercept)

    # Use ACTUAL observations used by each fit (NaN rows already dropped), not len()
    n_before = int(m_before.nobs)
    n_after  = int(m_after.nobs)
    n_total  = n_before + n_after

    df_num   = k
    df_denom = n_total - 2 * k

    f_stat = ((rss_p - (rss_b + rss_a)) / df_num) / ((rss_b + rss_a) / df_denom)
    p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)

    return {
        'breakpoint':  breakpoint,
        'f_statistic': f_stat,
        'p_value':     p_value,
        'df_num':      df_num,
        'df_denom':    df_denom,
        'n_before':    n_before,
        'n_after':     n_after,
        'break':       p_value < 0.05,
    }


def chow_test_multiple(df: pd.DataFrame, breakpoints: list,
                        formula: str) -> pd.DataFrame:
    """Run chow_test at each breakpoint; return a tidy DataFrame of results."""
    rows = []
    for bp in breakpoints:
        try:
            res = chow_test(df, bp, formula)
            rows.append(res)
        except Exception as e:
            # A sub-period may be too small to fit the model (esp. early eras).
            # Record the failure rather than crashing the whole loop.
            rows.append({
                'breakpoint':  bp,
                'f_statistic': np.nan,
                'p_value':     np.nan,
                'df_num':      np.nan,
                'df_denom':    np.nan,
                'n_before':    np.nan,
                'n_after':     np.nan,
                'break':       None,
                'error':       str(e),
            })

    return pd.DataFrame(rows).set_index('breakpoint')


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
    # grangercausalitytests expects columns ordered [effect, cause].
    # H0: `cause` does NOT Granger-cause `effect`.
    data = df[[effect, cause]].dropna()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # silences the per-lag verbose-deprecation noise
        results = grangercausalitytests(data, maxlag=maxlag, verbose=False)

    rows = []
    for lag in range(1, maxlag + 1):
        tests = results[lag][0]   # dict of test-name -> (stat, pvalue, ...)
        rows.append({
            'lag':         lag,
            'ssr_F':       tests['ssr_ftest'][0],
            'ssr_F_pval':  tests['ssr_ftest'][1],
            'ssr_chi2_pval': tests['ssr_chi2test'][1],
            'significant': tests['ssr_ftest'][1] < 0.05,
        })

    out = pd.DataFrame(rows).set_index('lag')
    out.attrs['hypothesis'] = f'{cause} Granger-causes {effect}'
    return out


def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """
    Engle-Granger cointegration test.

    H0: no cointegration. p < 0.05 → variables share a long-run
    equilibrium relationship even if both are non-stationary.

    Returns dict: statistic, p_value, critical_values.
    """
    # Align on the shared index and drop rows where either series is missing,
    # so both inputs cover exactly the same dates.
    joined = pd.concat([y, x], axis=1, keys=['y', 'x']).dropna()

    stat, p, crit = coint(joined['y'], joined['x'])

    return {
        'statistic':     stat,
        'p_value':       p,
        'crit_1%':       crit[0],
        'crit_5%':       crit[1],
        'crit_10%':      crit[2],
        'cointegrated':  p < 0.05,
        'n_obs':         len(joined),
    }


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def plot_series(df: pd.DataFrame, cols: list, log: bool = False,
                title: Optional[str] = None, halvings: bool = False):
    """Quick time-series plot of one or more columns, with optional log scale."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for col in cols:
        ax.plot(df.index, df[col], label=col, linewidth=1)

    if log:
        ax.set_yscale('log')

    # Optionally mark the halving dates with vertical lines — handy context
    if halvings:
        for d in HALVING_DATES:
            ax.axvline(pd.Timestamp(d), color='grey', linestyle='--',
                       alpha=0.5, linewidth=1)

    ax.set_xlabel('Date')
    ax.set_ylabel('Value' + (' (log scale)' if log else ''))
    if title:
        ax.set_title(title)
    if len(cols) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


def plot_residuals(model, lags: int = 40):
    """
    Diagnostic 2x2 panel for regression residuals:
    time series, histogram + KDE, Q-Q plot, ACF.

    Use this after every regression to check assumptions visually.
    """
    from statsmodels.graphics.tsaplots import plot_acf

    resid = model.resid

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) Residuals over time — look for changing variance / clustering
    axes[0, 0].plot(resid.index, resid, linewidth=0.8)
    axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.6)
    axes[0, 0].set_title('Residuals over time')
    axes[0, 0].set_xlabel('Date')

    # (2) Histogram + KDE — check for normality / skew / fat tails
    sns.histplot(resid, kde=True, ax=axes[0, 1], bins=60)
    axes[0, 1].set_title('Residual distribution')
    axes[0, 1].set_xlabel('Residual')

    # (3) Q-Q plot — diagnose tail behaviour against a normal
    sm.qqplot(resid, line='s', ax=axes[1, 0])
    axes[1, 0].set_title('Q-Q plot (vs normal)')

    # (4) ACF — check for leftover autocorrelation
    plot_acf(resid, lags=lags, ax=axes[1, 1])
    axes[1, 1].set_title('Residual autocorrelation (ACF)')

    plt.tight_layout()
    plt.show()
    return fig