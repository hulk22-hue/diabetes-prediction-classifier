import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_confusion_matrix(y_test, y_pred, model_name):
    ensure_dir('images')
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'images/confusion_matrix_{model_name}.png')
    plt.show()

def plot_roc_curve(y_test, y_pred_prob, model_name):
    ensure_dir('images')
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'images/roc_curve_{model_name}.png')
    plt.show()

def evaluate_and_compare(X_test, y_test, models):
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        plot_confusion_matrix(y_test, y_pred, model_name)

        plot_roc_curve(y_test, y_pred_prob, model_name)