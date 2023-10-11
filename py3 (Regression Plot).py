# %%
import pandas as pd

df = pd.read_csv("CAN-1985-2022.csv",
    index_col='dates',
    parse_dates=['dates'],
)

df.head()
# %%
import numpy as np

df['Time'] = np.arange(len(df.index))

df.head()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
%config InlineBackend.figure_format = 'retina'

fig, ax = plt.subplots()
ax.plot('year', 'electricity_generation', data=df, color='0.75')
ax = sns.regplot(x='year', y='electricity_generation', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Electricity Generation in Canada');

# %%
