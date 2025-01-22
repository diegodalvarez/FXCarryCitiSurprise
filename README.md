# FX Carry Citi Surprise


The overall goal of this repo is to build out a FX Carry strategy using the Citi Surprise indices as inputted values. For the time being carry will be proxied via the Deutsche Bank G10 carry index. A simple model can be created by conditioning on the raw Citi Surprise index. It can be enhanced by being long the residuals of an OLS model. With respect to the Citi indicators its not clear which ones to use in this case which Citi indicator to use. Instead the indices can be decomposed into their principal components and an individual model can be per each rank of the matrix.

|         | PDF          |
|----------------|---------------------|
| Technical Writeup containing methodology & results | <a href="https://github.com/diegodalvarez/FXCarryCitiSurprise/blob/main/FX_Carry_Citi_Surprise_Writeup.pdf"><img src="https://github.com/user-attachments/assets/1ac3065e-19be-4fc5-ab84-005af1758e8f" alt="image" width="500"/></a> |


# Todo
Bootstrap OLS model
