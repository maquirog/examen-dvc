
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data():
    scaler = MinMaxScaler()
    X_train = pd.read_csv("data/processed_data/X_train.csv")
    X_test = pd.read_csv("data/processed_data/X_test.csv")

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled

def to_csv_data(X_train_scaled, X_test_scaled):
    X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv', index=False)

def main():
    X_train_scaled, X_test_scaled = normalize_data()
    to_csv_data(X_train_scaled, X_test_scaled)


if __name__ == '__main__':
    main()