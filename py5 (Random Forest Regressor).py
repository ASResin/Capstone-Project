# %%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# %%

data = pd.read_csv('CAN-1985-2022.csv')

X = data['year'].values.reshape(-1, 1)
y = data['electricity_generation'].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)

# %%

predicted_electricity_2030_rf = rf_model.predict([[2030]])

print("Predicted Electricity Generation in 2030 Using Random Forest:", predicted_electricity_2030_rf)

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

rf_mae = mean_absolute_error(y_test, rf_y_pred)
print("Random Forest MAE:", rf_mae)

rf_mse = mean_squared_error(y_test, rf_y_pred)
print("Random Forest MSE:", rf_mse)

rf_rmse = np.sqrt(rf_mse)
print("Random Forest RMSE:", rf_rmse)

# %%
X1 = data['year'].values.reshape(-1, 1)
y1 = data['low_carbon_electricity'].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# %%
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)

# %%

predicted_low_carbon_2030_rf = rf_model.predict([[2030]])

print("Predicted Low Carbon Electricity Generation in 2030 Using Random Forest:", predicted_low_carbon_2030_rf)

# %%
X2 = data['year'].values.reshape(-1, 1)
y2 = data['renewables_electricity'].values

# %%
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# %%
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)

# %%

predicted_renewables_2030_rf = rf_model.predict([[2030]])

print("Predicted Renewable Electricity Generation in 2030 Using Random Forest:", predicted_renewables_2030_rf)

# %%

percentage_of_renewable_electricity_rf = (predicted_renewables_2030_rf / predicted_electricity_2030_rf) * 100
print("Percentage of Electricity Produced from Renewable Sources Using Random Forest: ",percentage_of_renewable_electricity_rf)

percentage_of_low_carbon_electricity_rf = (predicted_low_carbon_2030_rf / predicted_electricity_2030_rf) * 100
print("Percentage of Electricity Produced from Low Carbon Sources Using Random Forest: ",percentage_of_low_carbon_electricity_rf)
# %%
