import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_data(df):
    df = df.drop(columns='Unnamed: 32', errors='ignore')
    return df

def encode_categorical_data(df, column_name):
    try:
        df = pd.get_dummies(data=df, columns=[column_name], drop_first=True)
        df[f'{column_name}_M'] = df[f'{column_name}_M'].astype(int)
    except KeyError:
        print(f"Column {column_name} not found!")
    return df

def plot_count(data):
    plt.figure()
    sns.countplot(x=data)
    plt.show()

def plot_correlation(df, target_column):
    correlations = df.drop(columns=target_column).corrwith(df[target_column])
    plt.figure(figsize=(20,10))
    correlations.plot.bar(title='Correlation with Diagnosis', rot=45, grid=True)
    plt.show()

def plot_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(20,10))
    sns.heatmap(corr, annot=True)
    plt.show()

def split_data(df, target_column):
    X = df.drop(columns=target_column).values
    y = df[target_column].values
    return train_test_split(X, y, test_size=0.2, random_state=0)

def scale_features(X_train, X_test):
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_and_evaluate_model(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    return acc, f1, prec, rec, y_pred

def cross_validate_model(classifier, X_train, y_train):
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    mean_acc = accuracies.mean() * 100
    std_dev = accuracies.std() * 100
    return mean_acc, std_dev

if __name__ == "__main__":
    file_path = 'C:\\Users\\DELL\\Prog\\templates\\Data science projects\\breast-cancer-detection\\data\\data.csv'
    
    data = load_data(file_path)
    if data is not None:
        data = handle_missing_data(data)
        data = encode_categorical_data(data, 'diagnosis')
        
        plot_count(data['diagnosis_M'])
        plot_correlation(data, 'diagnosis_M')
        plot_heatmap(data)

        X_train, X_test, y_train, y_test = split_data(data, 'diagnosis_M')
        X_train, X_test = scale_features(X_train, X_test)
        
        # Logistic Regression
        classifier_lr = LogisticRegression(random_state=0)
        acc, f1, prec, rec, y_pred_lr = train_and_evaluate_model(classifier_lr, X_train, y_train, X_test, y_test)
        print(f"Logistic Regression Results: Accuracy={acc}, F1={f1}, Precision={prec}, Recall={rec}")
        mean_acc, std_dev = cross_validate_model(classifier_lr, X_train, y_train)
        print(f"Cross Validation Results for Logistic Regression: Mean Accuracy={mean_acc}, Standard Deviation={std_dev}")

        # Random Forest
        classifier_rm = RandomForestClassifier(random_state=0)
        acc, f1, prec, rec, y_pred_rf = train_and_evaluate_model(classifier_rm, X_train, y_train, X_test, y_test)
        print(f"Random Forest Results: Accuracy={acc}, F1={f1}, Precision={prec}, Recall={rec}")
        mean_acc, std_dev = cross_validate_model(classifier_rm, X_train, y_train)
        print(f"Cross Validation Results for Random Forest: Mean Accuracy={mean_acc}, Standard Deviation={std_dev}")
