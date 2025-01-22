# FX Carry Citi Surprise

## Approach
While investigating the Citi Economic Surprise Indices they proved themselves to be useful when trading  FX Carry (proxied by DB G10 carry index). This notebook is examining that relationship and will later expand to move away from the DB carry index and to specific carry indices. For the time being while the relationship is still being researcher the DB carry index will be the main proxy.

## Methodology
The initial approach was trading the carry index conditioned on the returns (i.e. ```np.sign(suprise_index) * index_rtn```). This proved surprisingly well for some specific surprise indices. In this case there is a macroeconomic explanation for why certain indices do bettr than others, for example G10 and US suprise indices while others countries don't. Even with that consideration most returns conditions on a Citi Surprise indices beat their benchmark. (It should be noticed that the benchmark doesn't return much overall, and returns pre-CIP (pre '08) are difficult to generate. 

![image](https://github.com/user-attachments/assets/5d847d89-36f3-469f-9605-ca4927f34a92)

Ideally rather than working with each Citi Surprise indices and then combining to get an overall portfolio a principal component approach can be used. They don't necessarily perform better than the raw conditioned returns. 
![image](https://github.com/user-attachments/assets/b80b9b9e-14ca-4fe3-9e3b-a3cbc6b84b60)

Now using a full-sample in-sample OLS and being long the residuals where the regression is the carry index regressed against the principal components gives

![image](https://github.com/user-attachments/assets/de2c5698-fd07-4ed5-bc6f-66ef4d97836a)

Each portfolios regresses the top $n$ PCs (i.e. the 10th regression is the $Rtn \sim \sum_{i = 1}^{10} \beta_i \cdot PC_i$ not $Rtn \sim \beta_{10} \cdot PC_{10}$) The previous graph is for the full rank of matrix. 

|         | PDF          |
|----------------|---------------------|
| Technical Writeup containing methodology & results | <a href="https://github.com/diegodalvarez/FXCarryCitiSurprise/blob/main/FX_Carry_Citi_Surprise_Writeup.pdf"><img src="https://github.com/user-attachments/assets/1ac3065e-19be-4fc5-ab84-005af1758e8f" alt="image" width="500"/></a> |


# Todo
1. Bootstrap OLS model
2. Make Model comparison
3. Raw Citi Surprise residual trading

