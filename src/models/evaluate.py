
import pickle
import json
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

X_test = pd.read_csv("data/processed_data/X_test.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")

with open("models/gbr_model.pkl", 'rb') as file:
    gbr = pickle.load(file)

y_pred = gbr.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
metrics = {"r2": r2, "mse": mse}

with open("metrics/scores.json", "w", encoding="utf8") as file:
    json.dump(metrics, file)
file.close()

print(y_pred)