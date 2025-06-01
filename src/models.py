from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_logistic_regression(X_train, y_train, random_state=42):
    """Trains a Logistic Regression model with balanced class weights."""
    model = LogisticRegression(solver='liblinear', random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier(X_train, y_train, random_state=42, n_estimators=100):
    """Trains a Random Forest Classifier with balanced class weights."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting_classifier(X_train, y_train, random_state=42, n_estimators=100, learning_rate=0.1):
    """Trains a Gradient Boosting Classifier."""
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_lightgbm_classifier(X_train, y_train, random_state=42, n_estimators=100, learning_rate=0.1):
    """Trains a LightGBM Classifier with balanced class weights."""
    model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_classification_model(model, X_test, y_test):
    """Evaluates the trained classification model and prints key metrics."""
    y_pred = model.predict(X_test)
    
    # Check if model supports predict_proba (required for AUC)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
        if y_pred_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            print(f"AUC-ROC: {auc:.4f}")
        else:
            print("AUC-ROC: Cannot compute for multiclass with one-vs-rest directly here.")
    else:
        y_pred_proba = None
        print("Model does not support predict_proba, skipping AUC.")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred, y_pred_proba
