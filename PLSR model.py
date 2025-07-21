# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  

# === read data ===
datatrain = pd.read_excel('***.xlsx', sheet_name='calibration', index_col=0)
datatest = pd.read_excel('***.xlsx', sheet_name='validation', index_col=0)

# Features and variables
X_train_features = datatrain.iloc[:, :-2].to_numpy()       
X_train_aux = datatrain.iloc[:, -2].to_numpy().reshape(-1, 1)  
y_train = datatrain.iloc[:, -1].to_numpy()               

X_test_features = datatest.iloc[:, :-2].to_numpy()
X_test_aux = datatest.iloc[:, -2].to_numpy().reshape(-1, 1)
y_test = datatest.iloc[:, -1].to_numpy()

feature_names = datatrain.columns[:-2].to_numpy()

# === Calculate Pearson correlation ===
correlations = np.array([np.corrcoef(X_train_features[:, i], y_train)[0, 1] for i in range(X_train_features.shape[1])])
abs_corr = np.abs(correlations)
sorted_idx = np.argsort(abs_corr)[::-1]
sorted_features = feature_names[sorted_idx]
sorted_correlations = correlations[sorted_idx]

# Save Pearson correlation sort results
pd.DataFrame({
    'Feature': sorted_features,
    'Pearson_Correlation': sorted_correlations
}).to_csv('pearson_correlation_sorted.csv', index=False)

# === Select the top 10% features ===
total_features = len(feature_names)
top_n = max(1, int(np.ceil(total_features * 0.1)))
top_features = sorted_features[:top_n]
top_indices = [np.where(feature_names == f)[0][0] for f in top_features]

# === input features ===
X_train_selected = X_train_features[:, top_indices]
X_test_selected = X_test_features[:, top_indices]

# Normalized  PV (0-1)
aux_scaler = MinMaxScaler()
X_train_aux_norm = aux_scaler.fit_transform(X_train_aux)
X_test_aux_norm = aux_scaler.transform(X_test_aux)

X_train_final = np.hstack([X_train_selected, X_train_aux_norm])
X_test_final = np.hstack([X_test_selected, X_test_aux_norm])

# Normalize all input features (PLSR recommends normalization)
x_scaler = StandardScaler()
X_train_std = x_scaler.fit_transform(X_train_final)
X_test_std = x_scaler.transform(X_test_final)

# === model calibration ===
n_components = min(15, X_train_std.shape[1])  
model = PLSRegression(n_components=n_components)
model.fit(X_train_std, y_train)

# === prediction ===
y_train_pred = model.predict(X_train_std).ravel()
y_test_pred = model.predict(X_test_std).ravel()

# === accuracy ===
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# === print result ===
print("\n calibration：")
print(f"RMSE: {train_rmse:.4f}")
print(f"R²:   {train_r2:.4f}")
print("\n📊 validation：")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")

# === save result ===
pd.DataFrame({
    'Selected_Features': list(top_features) + ['Auxiliary variables (normalized)']
}).to_csv('pls_top10_percent_with_aux_features.csv', index=False)

pd.DataFrame({
    'Observed': y_test,
    'Predicted': y_test_pred
}).to_csv('pls_top10_with_aux_predict_vs_true.csv', index=False)
