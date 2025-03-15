import joblib
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import pandas as pd

X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

with open("models/best_params.pkl", 'rb') as file:
    best_params = pickle.load(file)

print("best_params:", best_params)

gbr = GradientBoostingRegressor().fit(X_train_scaled, y_train)

with open("models/gbr_model.pkl", 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(gbr, file)
file.close()
