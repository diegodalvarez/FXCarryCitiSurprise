#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:39:43 2024

@author: diegoalvarez

"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from   sklearn.decomposition import PCA
from   FXCarryDataCollect import FXData

class PCAModel(FXData): 
    
    def __init__(self) -> None: 
        
        super().__init__()
        self.pca_data  = os.path.join(self.data_path, "PCAData")
        
        if os.path.exists(self.pca_data) == False: os.makedirs(self.pca_data)
        
    def _prep_data(self) -> pd.DataFrame: 
        
        df_surprise  = self.get_citi_suprise()
        good_tickers = (df_surprise.dropna().drop(
            columns = ["value"]).
            groupby("security").
            agg("count").
            assign(compare_value = lambda x: x.date.median() * 0.8).
            query("date >= compare_value").
            index.
            to_list())
        
        df_wider = (df_surprise.query(
            "security == @good_tickers").
            pivot(index = "date", columns = "security", values = "value").
            dropna())
        
        return df_wider
    
    def pca(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.pca_data, "CitiSurprisePCA.parquet")

        try:
            
            if verbose == True: print("Trying to find Citi Surprise PCA Return Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data now collecting it")
        
            df_wider  = self._prep_data()
            pca_model = PCA(n_components = len(df_wider.columns))
            
            df_fitted_values = (pd.DataFrame(
                data    = pca_model.fit_transform(df_wider),
                index   = df_wider.index,
                columns = ["PC{}".format(i + 1) for i in range(len(df_wider.columns))]).
                reset_index().
                melt(id_vars = "date").
                rename(columns = {"variable": "pc"}))
            
            df_var = (pd.DataFrame(
                data    = pca_model.fit(df_wider).explained_variance_ratio_,
                columns = ["explained_variance_ratio"],
                index   = ["PC{}".format(i + 1) for i in range(len(df_wider.columns))]).
                reset_index().
                rename(columns = {"index": "pc"}))
            
            df_out = (df_fitted_values.merge(
                right = df_var, how = "inner", on = ["pc"]))
            
            if verbose == True: print("Saving PCA data")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def full_sample_ols_index_backtest(self, verbose: bool = False) -> tuple: 
        
        signal_path = os.path.join(self.pca_data, "PCAIndexFullSampleOLSBacktestRtn.parquet")
        param_path  = os.path.join(self.pca_data, "PCAIndexFullSampleOLSBacktestParams.parquet")

        try:
            
            if verbose == True: print("Trying to find PCA OLS Model Data")
            df_signal = pd.read_parquet(path = signal_path, engine = "pyarrow")
            df_params = pd.read_parquet(path = param_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data now collecting it")
        
            df_pca = self.pca()
            df_rtn = (self.get_misc_indices().drop(
                columns = ["security", "clean_val", "raw_val"]).
                dropna())
            
            pcs = df_pca.pc.drop_duplicates().to_list()
            
            df_signal, df_params = pd.DataFrame(), pd.DataFrame()
            
            for i in range(1, len(pcs) + 1):
                
                pcs      = ["PC{}".format(j + 1) for j in range(i)]
                df_wider = (df_pca.query(
                    "pc == @pcs").
                    drop(columns = ["explained_variance_ratio"]).
                    pivot(index = "date", columns = "pc", values = "value").
                    merge(right = df_rtn, how = "inner", on = ["date"]).
                    set_index("date"))
                
                model = (sm.OLS(
                    endog = df_wider.rtn,
                    exog  = sm.add_constant(df_wider.drop(columns = ["rtn"]))).
                    fit())
                
                df_signal_tmp = (model.resid.to_frame(
                    name = "lag_resid").
                    shift().
                    dropna().
                    merge(right = df_rtn, how = "inner", on = ["date"]).
                    assign(pcs = i))
                
                df_param_tmp = (model.params.to_frame(
                    name = "param").
                    assign(
                        pvalue = model.pvalues,
                        pcs    = i))
                
                df_signal = pd.concat([df_signal, df_signal_tmp])
                df_params = pd.concat([df_params, df_param_tmp]) 
            
            if verbose == True: print("Saving data")
            df_signal.to_parquet(path = signal_path, engine = "pyarrow")
            df_params.to_parquet(path = param_path, engine = "pyarrow")
            
        return df_signal, df_params
        

            
def main() -> None: 
        
    PCAModel().pca(verbose = True)
    PCAModel().full_sample_ols_index_backtest(verbose = True)
    
if __name__ == "__main__": main()