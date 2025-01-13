# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:48:08 2025

@author: Diego
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
from CitiPCA import PCAModel
from sklearn.decomposition import PCA

class BootstrapOLS(PCAModel):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.bootstrap_path = os.path.join(self.data_path, "BootstrapOLS")
        if os.path.exists(self.bootstrap_path) == False: os.makedirs(self.bootstrap_path)
        
        self.sample_size = 0.15
        self.num_iters   = 3_000
        
    def prep_data(self) -> pd.DataFrame: 
        
        keep_values = "FXCARRSP"

        df_out = (self.get_misc_indices().query(
            "security == @keep_values").
            pivot(index = "date", columns = "security", values = "rtn").
            merge(right = self.pca(), how = "inner", on = ["date"]).
            drop(columns = ["explained_variance_ratio"]))
        
        return df_out
    
    def _bootstrap_ols(self, df: pd.DataFrame, sample_size: float) -> pd.DataFrame: 
        
        df_tmp = (df.set_index(
            "date").
            sample(frac = sample_size))
        
        df_params = (sm.OLS(
            endog = df_tmp.FXCARRSP,
            exog  = sm.add_constant(df_tmp.drop(columns = ["FXCARRSP"]))).
            fit().
            params.
            to_frame(name = "param").
            T.
            melt(id_vars = "const"))
        
        df_betas = (df_params.drop(
            columns = ["const"]).
            rename(columns = {
                "variable": "pc",
                "value"   : "param_value"}))
        
        alpha = df_params.const[0]
        
        df_out = (df.melt(
            id_vars = ["date", "FXCARRSP"]).
            merge(right = df_betas, how = "inner", on = ["pc"]).
            assign(tmp_value = lambda x: x.param_value * x.value)
            [["date", "FXCARRSP", "tmp_value"]].
            groupby(["date", "FXCARRSP"]).
            agg("sum").
            assign(predict = lambda x: x.tmp_value + alpha).
            reset_index().
            drop(columns = ["tmp_value"]).
            sort_values("date").
            assign(
                resid     = lambda x: x.predict - x.FXCARRSP,
                lag_resid = lambda x: x.resid.shift()).
            dropna())
        
        return df_out
    
    def bootstrap_ols(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.bootstrap_path, "BootstrappedOLSBacktest.parquet")
        try:
            
            if verbose == True: print("Looking for Bootstrapped OLS Model")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data collecting it now")    
        
            df_data = self.prep_data()
            df_out  = pd.DataFrame()
            pcs     = len(df_data.pc.drop_duplicates().sort_values().to_list())
            
            for i in range(pcs):
                
                tmp_pcs  = ["PC{}".format(j + 1) for j in range(i + 1)]
                df_wider = (df_data.query(
                    "pc == @tmp_pcs").
                    pivot(index = ["date", "FXCARRSP"], columns = "pc", values = "value").
                    reset_index())
                
                df_tmp = (pd.concat([
                    self._bootstrap_ols(df_wider, self.sample_size).assign(sim = k + 1, model = tmp_pcs[-1])
                    for k in tqdm(range(self.num_iters), desc = "Bootstrapping PC{} OLS".format(i + 1))]))
            
                df_out = pd.concat([df_out, df_tmp])
            
            if verbose == True: print("Saving data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_spef_sims(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_sharpe = (df[
            ["sim", "signal_rtn"]].
            groupby("sim").
            agg(["mean", "std"])
            ["signal_rtn"].
            rename(columns = {
                "mean": "mean_rtn",
                "std" : "std_rtn"}).
            assign(sharpe = lambda x: x.mean_rtn / x.std_rtn * np.sqrt(252)))
        
        data          = df_sharpe.sharpe.to_list()
        data          = np.sort(data)
        n             = len(data)
        median_sharpe = data[n // 2 - 1] if n % 2 == 0 else data[n // 2]
        
        df_out = (df_sharpe.query(
            "sharpe == @median_sharpe | sharpe == sharpe.min() | sharpe == sharpe.max()").
            sort_values("sharpe").
            assign(attribute = ["min_sharpe", "median_sharpe", "max_sharpe"]))
        
        return df_out
    
    def bootstrap_spef_sims(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.bootstrap_path, "SpefBootstrapSims.parquet")
        try:
            
            if verbose == True: print("Looking for specific bootstrap simulations")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data collecting it now") 
        
            df_out = (self.bootstrap_ols().assign(
                signal_rtn = lambda x: np.sign(x.lag_resid) * x.FXCARRSP).
                groupby("model").
                apply(self._get_spef_sims).
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

    def get_MaxMinMedian(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.bootstrap_path, "BootstrapMinMaxMedian.parquet")
        try:
            
            if verbose == True: print("Looking for MinMaxMedian bootstrap simulations")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find data collecting it now") 
        
            df_out = (self.bootstrap_spef_sims().merge(
                right = self.bootstrap_ols(), how = "inner", on = ["model", "sim"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
def main() -> None:
        
    df = BootstrapOLS().bootstrap_ols(verbose = True)
    df = BootstrapOLS().bootstrap_spef_sims(verbose = True)
    df = BootstrapOLS().get_MaxMinMedian(verbose = True)
    
if __name__ == "__main__": main()