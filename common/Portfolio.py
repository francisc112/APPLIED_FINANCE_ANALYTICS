
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import minimize


class Portfolio_Stats:

    def __init__(self):
        pass

    def annualize_rets(self,r, periods_per_year):
        """
        Annualizes a set of returns
        We should infer the periods per year
        but that is currently left as an exercise
        to the reader :-)
        """
        compounded_growth = (1+r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(periods_per_year/n_periods)-1

    def skewness(self,r):
        """
        Alternative to scipy.stats.skew()
        Computes the skewness of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_r = r - r.mean()
        # use the population standard deviation, so set dof=0
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**3).mean()
        return exp/sigma_r**3


    def kurtosis(self,r):
        """
        Alternative to scipy.stats.kurtosis()
        Computes the kurtosis of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_r = r - r.mean()
        # use the population standard deviation, so set dof=0
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**4).mean()
        return exp/sigma_r**4

    def annualize_vol(self,r, periods_per_year):
        """
        Annualizes the vol of a set of returns
        We should infer the periods per year
        but that is currently left as an exercise
        to the reader :-)
        """
        return r.std()*(periods_per_year**0.5)

    def annualize_var(self,r, periods_per_year):
        """
        Annualizes the variance of a set of returns
        We should infer the periods per year
        but that is currently left as an exercise
        to the reader :-)
        """
        return r.var()*(periods_per_year)

    def sharpe_ratio(self,r, riskfree_rate, periods_per_year):
        """
        Computes the annualized sharpe ratio of a set of returns
        """
        # convert the annual riskfree rate to per period
        rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
        excess_ret = r - rf_per_period
        ann_ex_ret = self.annualize_rets(excess_ret, periods_per_year)
        ann_vol = self.annualize_vol(r, periods_per_year)
        return ann_ex_ret/ann_vol
    
    def drawdown(self,return_series: pd.Series):
        """Takes a time series of asset returns.
        returns a DataFrame with columns for
        the wealth index, 
        the previous peaks, and 
        the percentage drawdown
        """
        wealth_index = 1000*(1+return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks
        return pd.DataFrame({"Wealth": wealth_index, 
                            "Previous Peak": previous_peaks, 
                            "Drawdown": drawdowns})
    
    def semideviation(self,r):
        """
        Returns the semideviation aka negative semideviation of r
        r must be a Series or a DataFrame, else raises a TypeError
        """
        if isinstance(r, pd.Series):
            is_negative = r < 0
            return r[is_negative].std(ddof=0)
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(self.semideviation)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    def var_gaussian(self,r, level=5, modified=False):
        """
        Returns the Parametric Gauusian VaR of a Series or DataFrame
        If "modified" is True, then the modified VaR is returned,
        using the Cornish-Fisher modification
        """
        # compute the Z score assuming it was Gaussian
        z = norm.ppf(level/100)
        if modified:
            # modify the Z score based on observed skewness and kurtosis
            s = self.skewness(r)
            k = self.kurtosis(r)
            z = (z +
                    (z**2 - 1)*s/6 +
                    (z**3 -3*z)*(k-3)/24 -
                    (2*z**3 - 5*z)*(s**2)/36
                )
        return -(r.mean() + z*r.std(ddof=0))
    
    def cvar_historic(self,r, level=5):
        """
        Computes the Conditional VaR of Series or DataFrame
        """
        if isinstance(r, pd.Series):
            is_beyond = r <= - self.var_historic(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(self.cvar_historic, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")


    def summary_stats(self,r, riskfree_rate=0.03,periods_per_year = 365):
        """
        Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
        """
        ann_r = r.aggregate(self.annualize_rets, periods_per_year=periods_per_year)
        ann_vol = r.aggregate(self.annualize_vol, periods_per_year=periods_per_year)
        ann_semideviation = r.aggregate(self.semideviation) * (periods_per_year ** 0.5)
        sharpe_ratio = r.aggregate(self.sharpe_ratio,riskfree_rate=riskfree_rate,periods_per_year=periods_per_year)
        ann_sr = r.aggregate(self.sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year)
        dd = r.aggregate(lambda r: self.drawdown(r).Drawdown.min())
        skew = r.aggregate(self.skewness)
        kurt = r.aggregate(self.kurtosis)
        cf_var5 = r.aggregate(self.var_gaussian, modified=True)
        hist_cvar5 = r.aggregate(self.cvar_historic)
        return pd.DataFrame({
            "Annualized Return": ann_r,
            "Annualized Vol": ann_vol,
            "Annaulized Semideviation":ann_semideviation,
            "Sharpe Ratio":sharpe_ratio,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Cornish-Fisher VaR (5%)": cf_var5,
            "Historic CVaR (5%)": hist_cvar5,
            "Sharpe Ratio": ann_sr,
            "Max Drawdown": dd
        })

    def var_historic(self,r, level=5):
        """
        Returns the historic Value at Risk at a specified level
        i.e. returns the number such that "level" percent of the returns
        fall below that number, and the (100-level) percent are above
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(self.var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    def portfolio_return(self,weights, returns):
        """
        Computes the return on a portfolio from constituent returns and weights
        weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
        """
        return weights.T @ returns


    def portfolio_vol(self,weights, covmat):
        """
        Computes the vol of a portfolio from a covariance matrix and constituent weights
        weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
        """
        return (weights.T @ covmat @ weights)**0.5
    
    def semideviation(self,r):
        """
        Returns the semideviation aka negative semideviation of r
        r must be a Series or a DataFrame, else raises a TypeError
        """
        if isinstance(r, pd.Series):
            is_negative = r < 0
            return r[is_negative].std(ddof=0)
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(self.semideviation)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
        
    def sharpe_ratio(self,r, riskfree_rate, periods_per_year):
        """
        Computes the annualized sharpe ratio of a set of returns
        """
        # convert the annual riskfree rate to per period
        rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
        excess_ret = r - rf_per_period
        ann_ex_ret = self.annualize_rets(excess_ret, periods_per_year)
        ann_vol = self.semideviation(r)*(periods_per_year**0.5)
        return ann_ex_ret/ann_vol
    
    def minimize_vol(self,target_return, er, cov):
        """
        Returns the optimal weights that achieve the target return
        given a set of expected returns and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        return_is_target = {'type': 'eq',
                            'args': (er,),
                            'fun': lambda weights, er: target_return - self.portfolio_return(weights,er)
        }
        weights = minimize(self.portfolio_vol, init_guess,
                        args=(cov,), method='SLSQP',
                        options={'disp': False},
                        constraints=(weights_sum_to_1,return_is_target),
                        bounds=bounds)
        return weights.x
    
    def msr(self,riskfree_rate, er, cov):
        """
        Returns the weights of the portfolio that gives you the maximum sharpe ratio
        given the riskfree rate and expected returns and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        def neg_sharpe(weights, riskfree_rate, er, cov):
            """
            Returns the negative of the sharpe ratio
            of the given portfolio
            """
            r = self.portfolio_return(weights, er)
            vol = self.portfolio_vol(weights, cov)
            return -(r - riskfree_rate)/vol
        
        weights = minimize(neg_sharpe, init_guess,
                        args=(riskfree_rate, er, cov), method='SLSQP',
                        options={'disp': False},
                        constraints=(weights_sum_to_1,),
                        bounds=bounds)
        return weights.x
    
    def gmv(self,cov):
        """
        Returns the weights of the Global Minimum Volatility portfolio
        given a covariance matrix
        """
        n = cov.shape[0]
        return self.msr(0, np.repeat(1, n), cov)
    

    def optimal_weights(self,n_points, er, cov):
        """
        Returns a list of weights that represent a grid of n_points on the efficient frontier
        """
        target_rs = np.linspace(er.min(), er.max(), n_points)
        weights = [self.minimize_vol(target_return, er, cov) for target_return in target_rs]
        return weights
    
    def plot_ef(self,n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
        """
        Plots the multi-asset efficient frontier
        """
        weights = self.optimal_weights(n_points, er, cov)
        rets = [self.portfolio_return(w, er) for w in weights]
        vols = [self.portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols
        })
        ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
        if show_cml:
            ax.set_xlim(left = 0)
            # get MSR
            w_msr = self.msr(riskfree_rate, er, cov)
            r_msr = self.portfolio_return(w_msr, er)
            vol_msr = self.portfolio_vol(w_msr, cov)
            # add CML
            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate, r_msr]
            ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
        if show_ew:
            n = er.shape[0]
            w_ew = np.repeat(1/n, n)
            r_ew = self.portfolio_return(w_ew, er)
            vol_ew = self.portfolio_vol(w_ew, cov)
            # add EW
            ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
        if show_gmv:
            w_gmv = self.gmv(cov)
            r_gmv = self.portfolio_return(w_gmv, er)
            vol_gmv = self.portfolio_vol(w_gmv, cov)
            # add EW
            ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
            
            return ax
