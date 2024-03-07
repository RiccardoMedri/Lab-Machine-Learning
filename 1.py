import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

'''fromInternat = True
if fromInternat:
    os.environ['KAGGLE_USERNAME'] = 'riccardomedri'
    os.environ['KAGGLE_KEY'] = 'c8e22e7b961a65aee0e66ae9f00b49c1'
    from kaggle.api.kaggle_api_extended import KaggleApi
    dataset = 'mustafaali96/weight-height'
    path = 'datasets/heightsweights'
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path)
    api.dataset_download_file(dataset, 'weight-height.csv', path)'''

df = pd.read_csv('datasets/heightsweights/weight-height.csv')
'''head = df.head()
print(df[['Weight']].mean())
print(np.std(df['Weight']))'''
x = df.iloc[:,1]*2.54
y = df.iloc[:,2]*0.553
x1 = sm.add_constant(x)
model = sm.OLS(y,x1)
results = model.fit()
print(results.summary())
print("Parameters: ", results.params)
print("R2: ", results.rsquared)
pred = results.get_prediction()
ivlow = pred.summary_frame()["obs_ci_lower"] # lower interval
ivup = pred.summary_frame()["obs_ci_upper"] # upper interval
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, "o", label="data")
ax.plot(x, results.fittedvalues, "r--.", label="OLS")
ax.plot(x, ivup, "r--"); ax.plot(x, ivlow, "r--")
ax.legend(loc="best"); plt.xlabel("cm"); plt.ylabel("Kg")
plt.show()
p = np.corrcoef(x,y) # pearson xx, xy, yx, yy
print(f"Numpy, Pearson = {p}")
r = pd.Series(x).corr(y)
rho = x.corr(y, method='spearman')
tau = x.corr(y, method='kendall')
print(f"Pandas: pearson: {r} spearman: {rho} kendall: {tau}")