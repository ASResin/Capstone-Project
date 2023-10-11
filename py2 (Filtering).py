# %%
import numpy as np
import pandas as pd


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("OWID_Energy_CAN.csv")
df.head()

# %%
df['dates'] = pd.to_datetime(df['year'], format='%Y')

# %%
df = df[df['dates'] > '1984-01-01']

#%%
def plot_df(
    df,
    x,
    y,
    title="",
    xlabel='Year', 
    ylabel='Electricity Generated', 
    dpi=100):
        plt.figure(figsize=(15,4), dpi=dpi)
        plt.plot(x, y, color='tab:red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()
    

plot_df(df, x=df['year'], y=df['electricity_generation'], title='Electricity Generated in Canada')
# %%
from statsmodels.tsa.stattools import grangercausalitytests
data = df
grangercausalitytests(data[['electricity_generation', 'year']], maxlag=2)

# %%
df.to_csv("CAN-1985-2022.csv")
# %%
