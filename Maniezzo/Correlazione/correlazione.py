import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


#Dataset contenente valori e informazioni su immobili venduti in Canada (origine Kaggle)

df = pd.read_csv('houses/houses.csv', delimiter=';')

#L'obiettivo di analisi è calcolare la relazione fra superficie della proprietà e costo della stessa

#Escludo gli outlier, per entrambe le colonne, con il metodo dello z-score
threshold = 3
z_scores = (df['Sq.Ft'] - df['Sq.Ft'].mean()) / df['Sq.Ft'].std()
df = df[(z_scores < threshold) & (z_scores > -threshold)]
z_scores_price = (df['Price'] - df['Price'].mean()) / df['Price'].std()
df = df[(z_scores_price < threshold) & (z_scores_price > -threshold)]

#Concentro il dataset alle colonne di interesse ed elimino possibili valori fuorvianti (nulli o inf)
colonne = ['Sq.Ft', 'Price']
df = df[colonne]
df = df.dropna(subset=['Sq.Ft', 'Price'], how='any')
df = df[~df['Sq.Ft'].isin([np.inf, -np.inf])]
df = df[~df['Price'].isin([np.inf, -np.inf])]

x = df.iloc[:,0]
y = df.iloc[:,1]
x1 = sm.add_constant(x)
model = sm.OLS(y,x1)
results = model.fit()
print(results.summary())
print("Parameters: ", results.params)
print("R2: ", results.rsquared)
pred = results.get_prediction()
ivlow = pred.summary_frame()["obs_ci_lower"]
ivup = pred.summary_frame()["obs_ci_upper"]
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, "o", label="data")
ax.plot(x, results.fittedvalues, "r--.", label="OLS")
ax.plot(x, ivup, "r--"); ax.plot(x, ivlow, "r--")
ax.legend(loc="best"); plt.xlabel("Square Feet"); plt.ylabel("Canadian Dollar")
ax.yaxis.set_major_formatter('{x:,.0f}')
plt.show()
p = np.corrcoef(x,y)
print(f"Numpy, Pearson = {p}")
r = pd.Series(x).corr(y)
rho = x.corr(y, method='spearman')
tau = x.corr(y, method='kendall')
print(f"Pandas: pearson: {r} spearman: {rho} kendall: {tau}")