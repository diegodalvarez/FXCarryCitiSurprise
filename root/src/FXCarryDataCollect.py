#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:04:06 2024

@author: diegoalvarez
"""
import os
import numpy as np
import pandas as pd

class FXData:

    def __init__(self) -> None: 
        
        self.root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.repo_path = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.data_path = os.path.join(self.repo_path, "data")
        self.raw_path  = os.path.join(self.data_path, "RawData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_path)  == False: os.makedirs(self.raw_path)
        
        self.bbg_raw_path = r"/Users/diegoalvarez/Desktop/BBGData"
        if os.path.exists(self.bbg_raw_path) == False: 
            self.bbg_raw_path = r"C:\Users\Diego\Desktop\app_prod\BBGData"
        
        self.bbg_ticker_path = os.path.join(self.bbg_raw_path, "root", "BBGTickers.xlsx")
        self.df_tickers      = (pd.read_excel(io = self.bbg_ticker_path))
        
    
    def get_carry_return(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.raw_path, "FXCarryReturn.parquet")

        try:
            
            if verbose == True: print("Trying to find FX Carry Return Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data now collecting it")
        
            tickers = (self.df_tickers.query(
                "Category == 'Currencies' & Subcategory == 'Return Indices'").
                assign(Security = lambda x: x.Security.str.split(" ").str[0]).
                Security.
                to_list())
            
            paths = [
                os.path.join(self.bbg_raw_path, "data", ticker + ".parquet")
                for ticker in tickers]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                assign(
                    date     = lambda x: pd.to_datetime(x.date).dt.date,
                    security = lambda x: x.security.str.split(" ").str[0]).
                drop(columns = ["variable"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_citi_suprise(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "CitiSurprise.parquet")

        try:
            
            if verbose == True: print("Trying to find Citi Surprise Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data now collecting it")
    
            tickers = (self.df_tickers.query(
                "Category == 'Index/Stats' & Subcategory == 'Equity Index'").
                assign(
                    Security = lambda x: x.Security.str.split(" ").str[0],
                    ending   = lambda x: x.Description.str.split(" ").str[0]).
                query("ending == 'Citi'").
                Security.
                drop_duplicates().
                to_list())
            
            paths = [
                os.path.join(self.bbg_raw_path, "data", ticker + ".parquet")
                for ticker in tickers]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                assign(
                    security = lambda x: x.security.str.split(" ").str[0],
                    date     = lambda x: pd.to_datetime(x.date).dt.date).
                drop(columns = ["variable"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_misc_indices(self, verbose: bool = False) -> pd.DataFrame:
        
        
        file_path = os.path.join(self.raw_path, "MiscIndices.parquet")

        try:
            
            if verbose == True: print("Trying to find Citi Surprise Data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find data now collecting it")
        
            tickers = ["FXCARRSP"]
            paths   = [
                os.path.join(self.bbg_raw_path, "data", ticker + ".parquet")
                for ticker in tickers]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                assign(
                    security = lambda x: x.security.str.split(" ").str[0],
                    date     = lambda x: pd.to_datetime(x.date).dt.date).
                groupby("security").
                apply(self._double_clean).
                drop(columns = ["security"]).
                reset_index().
                drop(columns = ["first_clean"]).
                rename(columns = {
                    "second_clean": "clean_val",
                    "raw_value"   : "raw_val"}))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _double_clean(
            self,
            df           : pd.DataFrame,
            long_replace : int = 20, 
            long_window  : int = 200, 
            short_replace: int = 5, 
            short_window : int = 30) -> pd.DataFrame:
    
        df_out = (df.
            assign(security = lambda x: x.security.str.split(" ").str[0]).
            assign(
                short_mean  = lambda x: x.value.rolling(window = 20).mean(),
                roll_mean   = lambda x: x.value.rolling(window = 200).mean(),
                roll_std    = lambda x: x.value.rolling(window = 200).std(),
                z_score     = lambda x: np.abs((x.value - x.roll_mean) / x.roll_std),
                first_clean = lambda x: np.where(x.z_score > 4, x.short_mean, x.value)).
            set_index("date")
            [["value", "first_clean", "security"]].
            assign(
                short_mean  = lambda x: x.first_clean.rolling(window = 5).mean(),
                roll_mean   = lambda x: x.first_clean.rolling(window = 30).mean(),
                roll_std    = lambda x: x.first_clean.rolling(window = 30).std(),
                z_score     = lambda x: np.abs((x.value - x.roll_mean) / x.roll_std),
                second_clean = lambda x: np.where(x.z_score > 2.5, x.short_mean, x.first_clean))
            [["value", "security", "first_clean", "second_clean"]].
            rename(columns = {"value": "raw_value"}).
            sort_index().
            assign(rtn = lambda x: x.second_clean.pct_change()))
        
        return df_out
  
def main() -> None:       
        
    df = FXData().get_carry_return(verbose = True)
    df = FXData().get_citi_suprise(verbose = True)
    df = FXData().get_misc_indices(verbose = True)
    
if __name__ == "__main__": main()