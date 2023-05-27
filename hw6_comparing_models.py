import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class ClassifierComparison:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.data_test = None
        self.target_test = None
        self.target = None
        self.model_lr = None
        self.model_knn = None
        self.model_mlp = None

    def load_data(self):
        df = pd.read_csv(self.dataset_path)

        df1 = df.copy(deep=True)  # making a copy of the dataframe to protect original data

        # define the columns to be encoded and scaled
        categorical_columns = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
        continious_columns = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

        # encoding the categorical columns
        df1 = pd.get_dummies(df1, columns=categorical_columns, drop_first=True)
        # %%

        # # defining the features and target
        X = df1.drop(['output'], axis=1)
        y = df1[['output']]

        # # instantiating the scaler
        scaler = RobustScaler()

        # # scaling     the continuous featuree
        X[continious_columns] = scaler.fit_transform(
            X[continious_columns])  # Transform the continious column to have unit variance and zero mean
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.data = X_train
        self.target = y_train
        self.data_test = X_test
        self.target_test = y_test

    def train_models(self):
        self.model_lr = LogisticRegression()
        self.model_lr.fit(self.data, self.target)

        self.model_knn = KNeighborsClassifier()
        self.model_knn.fit(self.data, self.target)

        self.model_mlp = MLPClassifier()
        self.model_mlp.fit(self.data.astype(float), self.target)

    def predict(self):
        lr_predictions = self.model_lr.predict(self.data_test)
        knn_predictions = self.model_knn.predict(self.data_test)
        mlp_predictions = self.model_mlp.predict(self.data_test)

        return lr_predictions, knn_predictions, mlp_predictions

    def compare_metrics(self):
        lr_predictions, knn_predictions, mlp_predictions = self.predict()

        lr_accuracy = accuracy_score(self.target_test, lr_predictions)
        knn_accuracy = accuracy_score(self.target_test, knn_predictions)
        mlp_accuracy = accuracy_score(self.target_test, mlp_predictions)

        print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
        print(f"KNN Accuracy: {knn_accuracy:.4f}")
        print(f"MLP Accuracy: {mlp_accuracy:.4f}")

    def plot_roc_auc_curves(self):
        lr_probabilities = self.model_lr.predict_proba(self.data_test)[:, 1]
        knn_probabilities = self.model_knn.predict_proba(self.data_test)[:, 1]
        mlp_probabilities = self.model_mlp.predict_proba(self.data_test)[:, 1]

        lr_auc = roc_auc_score(self.target_test, lr_probabilities)
        knn_auc = roc_auc_score(self.target_test, knn_probabilities)
        mlp_auc = roc_auc_score(self.target_test, mlp_probabilities)

        fpr_lr, tpr_lr, _ = roc_curve(self.target_test, lr_probabilities)
        fpr_knn, tpr_knn, _ = roc_curve(self.target_test, knn_probabilities)
        fpr_mlp, tpr_mlp, _ = roc_curve(self.target_test, mlp_probabilities)
        plt.figure(figsize=(6, 3))
        plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_auc:.2f})")
        plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {knn_auc:.2f})")
        plt.plot(fpr_mlp, tpr_mlp, label=f"MLP (AUC = {mlp_auc:.2f})")

        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.tight_layout()
        plt.show()


# Usage Example
cc = ClassifierComparison(Path('/Users/anmolgorakshakar/Downloads/heart.csv'))
cc.load_data()
cc.train_models()


cc.compare_metrics()
cc.plot_roc_auc_curves()
