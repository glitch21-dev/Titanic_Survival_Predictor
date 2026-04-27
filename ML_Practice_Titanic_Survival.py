import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Titanic-Dataset.csv')
data.info()
print(data.isnull().sum()) #isnull()tells us the missing values (returns a boolean) and sum() returns the sum of our missing values

#some datatypes need to be converted to usable formats like numbers e.g gender

#Data Cleaming Phase
def preprocess_data(df):
    df = df.copy()
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)#specifying which columns to drop, the non-essential ones and replacing with true/ false

    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Embarked"], inplace=True)

    fill_missing_ages(df)

    #convert the gender to boolean
    df["Sex"] = df["Sex"].map({"male":1, "female":0}) #here we have converted male to 1 and female to 0

    #Feature engineering- creating new columns in our data e.g family size by combining siblings and parents
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1,0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins= [0, 12, 20, 40, 60, np.inf], labels=False)

    return df


#Filling the missing ages - this funtion will continuously fill in the missng values
def fill_missing_ages(df):
    #we can try to use a median for all ages to guess the age of the missing values
    #we can create a dicktionary for this
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass] ["Age"].median() #anywhere where pclass==age, we get the median() python handles the implementation

  
     #corrected funtion  

    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)


data = preprocess_data(data)

    #Creating Features and Target Variables (more like flashcards - the model is allowed to look at both the training data and testing data)

    
x = data.drop(columns=["Survived"])
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    #ML Preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

    #Hyperparameter Tuning - KNN
def tune_model(X_train, y_train):  #we create a dictionary full of our key value pairs here
        param_grid = { 
            "n_neighbors": range(1,21),
            "metric" : ["euclidean", "manhattan", "minkowski"],
            "weights" : ["uniform", "distance"]
        }

        model = KNeighborsClassifier()
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
best_model = tune_model(X_train, y_train)


    #Predictions and Evaluation
def evaluate_model(model, X_test, y_test):
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        matrix = confusion_matrix(y_test, prediction)
        return accuracy, matrix
    
accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f"Confusion Matrix: ")
print(matrix)

    #Plot
def plot_model(matrix):
        plt.figure(figsize=(10,7))
        sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived", "Not Survived"], yticklabels=["Not Survived", "Survived"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Value")
        plt.ylabel("True Values")
        plt.show()

plot_model(matrix)
