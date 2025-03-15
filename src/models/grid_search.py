from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import pandas as pd

X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

model = GradientBoostingRegressor()
grid = dict()
grid['n_estimators'] = [100, 500]
grid['learning_rate'] = [0.01, 0.1]
grid['subsample'] = [0.5, 1.0]
grid['max_depth'] = [3, 7, 9]
# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=2, n_jobs=-1)
# execute the grid search
grid_result = grid_search.fit(X_train_scaled, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

data = grid_result.best_params_
with open("models/best_params.pkl", 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(data, file)
file.close()
