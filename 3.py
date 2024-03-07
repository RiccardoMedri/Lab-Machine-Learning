import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
column = "Height" # Height
df=pd.read_csv('datasets/heightsweights/weight-height.csv',usecols=[column])
# histogram plot
plt.hist(df[column])
# QQ plot
print(f'QQ plot of: {column}')
sm.qqplot(df[column], line='s')
plt.show()
# Anderson-Darling normality test
ad,p = sm.stats.diagnostic.normal_ad(df[column])
print(f"Anderson-Darling {ad}, p={p}")