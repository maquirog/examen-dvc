
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(df):
    target = df['silica_concentrate']
    feats = df.drop(['date','silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def to_csv_data(X_train, X_test, y_train, y_test):

    os.makedirs("data/processed_data")

    X_train.to_csv('data/processed_data/X_train.csv', index=False)
    X_test.to_csv('data/processed_data/X_test.csv', index=False)
    y_train.to_csv('data/processed_data/y_train.csv', index=False)
    y_test.to_csv('data/processed_data/y_test.csv', index=False)

def main():
    df = pd.read_csv("data/raw_data/raw.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    to_csv_data(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()