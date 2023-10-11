# %% import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%

data = pd.read_csv('CAN-1985-2022.csv')

X = data['year'].values.reshape(-1, 1)
y = data['electricity_generation'].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%

model = LinearRegression()
model.fit(X_train, y_train)

# %%

predicted_energy_2030 = model.predict([[2030]])
print("Predicted Electricity Generation in 2030:", predicted_energy_2030)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# %%
X1 = data['year'].values.reshape(-1, 1)
y1 = data['low_carbon_electricity'].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# %%

model = LinearRegression()
model.fit(X_train, y_train)

# %%

predicted_low_carbon_2030 = model.predict([[2030]])
print("Predicted Low Carbon Electricity Generation in 2030:", predicted_low_carbon_2030)

# %% 
X2 = data['year'].values.reshape(-1, 1)
y2 = data['renewables_electricity'].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# %%

model = LinearRegression()
model.fit(X_train, y_train)

# %%

predicted_renewables_2030 = model.predict([[2030]])
print("Predicted Renewable Electricity Generation in 2030:", predicted_renewables_2030)

# %%

percentage_of_renewable_electricity = (predicted_renewables_2030 / predicted_energy_2030) * 100
print("Percentage of Electricity Produced from Renewable Sources: ",percentage_of_renewable_electricity)

percentage_of_low_carbon_electricity = (predicted_low_carbon_2030 / predicted_energy_2030) * 100
print("Percentage of Electricity Produced from Low Carbon Sources: ",percentage_of_low_carbon_electricity)
# %%
