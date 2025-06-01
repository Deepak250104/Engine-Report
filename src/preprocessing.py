import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split_data(data_path, test_size=0.2, random_state=42, stratify_col='Engine Condition'):
    """Loads data and splits it into training and testing sets, stratified by the target variable."""
    df = pd.read_csv(data_path)
    X = df.drop(stratify_col, axis=1)
    y = df[stratify_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """Scales the training and testing data using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
