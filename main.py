from src import (
    load_and_preprocess_data, 
    train_logistic_regression, 
    train_svm, 
    train_random_forest, 
    evaluate_and_compare
)

X_train, X_test, y_train, y_test = load_and_preprocess_data('data/pima-indians-diabetes.csv')

log_reg_model = train_logistic_regression(X_train, X_test, y_train, y_test)

svm_model = train_svm(X_train, X_test, y_train, y_test)

rf_model = train_random_forest(X_train, X_test, y_train, y_test)

models = {
    "Logistic Regression": log_reg_model,
    "SVM": svm_model,
    "Random Forest": rf_model
}

evaluate_and_compare(X_test, y_test, models)