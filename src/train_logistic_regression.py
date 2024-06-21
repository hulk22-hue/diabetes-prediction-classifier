from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)

    print("Logistic Regression Classification Report")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    return log_reg