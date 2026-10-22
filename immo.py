import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Chargement des données
data = fetch_california_housing(as_frame=True)
df = data.frame
# Préparation
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Modèle (PoC)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
# Évaluation
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"MAE obtenue : {mae:.4f}") # Doit être < 0.40