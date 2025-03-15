from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df):
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    df = pd.read_csv("data/raw_data/raw.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

if __name__ == '__main__':
    main()