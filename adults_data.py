import pandas as pd
import sklearn
from sklearn import preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv("adult.data")
dataFrame = df.replace("?", -99999, inplace=True)

print(df)
print(dataFrame)

collumnLabel = preprocessing.LabelEncoder()


sex = collumnLabel.fit_transform(df["sex"])
age = df["age"]
workclass = collumnLabel.fit_transform(df["workclass"])
fnlwgt = df["fnlwgt"]
education = collumnLabel.fit_transform(df["education"])
education_num = df["education-num"]
marital_status = collumnLabel.fit_transform(df["marital-status"])
occupation = collumnLabel.fit_transform(df["occupation"])
relationship = collumnLabel.fit_transform(df["relationship"])
race = collumnLabel.fit_transform(df["race"])
capital = df["capital"]
capital_loss = df["capital-loss"]
hours_per_week = df["hours-per-week"]
native_country = collumnLabel.fit_transform(df["native-country"])

predict = "salary"
mysalary = collumnLabel.fit_transform(df[predict])

x = list(zip(sex, age, workclass, fnlwgt, education, education_num, occupation, marital_status, relationship, race, capital, capital_loss, hours_per_week, native_country))
y = list(mysalary)

def trainLinearModel():
    x_test, x_train, y_test, y_train = sklearn.model_selection.train_test_split(x, y, test_size=0.15)

    for _ in range(30):
        linearModel = linear_model.LinearRegression()

        linearModel.fit(x_train, y_train)

        acc = linearModel.score(x_test, y_test)
        print(acc)

x_test, x_train, y_test, y_train = sklearn.model_selection.train_test_split(x, y, test_size=0.15)

def trainNeighborsModel():
    #x_test, x_train, y_test, y_train = sklearn.model_selection.train_test_split(x, y, test_size=0.10)

    neigborModel = KNeighborsClassifier(n_neighbors=13)

    neigborModel.fit(x_train, y_train)

    acc = neigborModel.score(x_test, y_test)
    best_score = 0.79

    if acc > best_score:
        print(acc)
        with open("adult-naighbors.pickle", "wb") as f:
            pickle.dump(neigborModel, f)

def runNeighborsModel():
    modelData = open("adult-naighbors.pickle", "rb")
    ourModel = pickle.load(modelData)

    prediction = ourModel.predict(x_test)
    names = ["<=50K", ">50K"]
    print(len(x_test))
    
    for x in range(len(x_test)):
        print("Our Prediction: ", names[prediction[x]], " ---> Actual Value: ", names[y_test[x]])


    '''for x in range(len(x_test)):
        print("Predicted: ", names[prediction[x]], " Actual Value: ", names[y_test[x]])'''

for _ in range(30):
    trainNeighborsModel()

#runNeighborsModel()