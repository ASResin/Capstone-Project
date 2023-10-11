# %% Importing Libraries
import pandas as pd

df = pd.read_csv("owid-energy-data.csv")

# %% Choosing Countries to Keep
countries_to_keep = ["CAN" , "USA" , "GBR", "AUS"]
mask = df['iso_code'].isin(countries_to_keep)

df = df[mask]

# %% Choosing Years to Keep
df['year'] = df['year'].astype(int)
df = df[df['year'] >= 1970]

# %% resetting the index
df = df.reset_index(drop=True)

# %% Replacing Blanks with 0
df = df.fillna(0)

# %%
df.to_csv("Clean_OWID_Energy (All Countries).csv")
 
# %%
