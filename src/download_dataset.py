import pandas as pd
import os

def download_and_save_data(url, file_path):
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    data = pd.read_csv(url, header=None, names=column_names)

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    data.to_csv(file_path, index=False)
    print(f"Dataset downloaded and saved as '{file_path}'")

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    file_path = os.path.join('..', 'data', 'pima-indians-diabetes.csv')
    
    download_and_save_data(url, file_path)