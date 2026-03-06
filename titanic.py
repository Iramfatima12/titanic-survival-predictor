import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


df=pd.read_csv("train.csv")

#display data 
# print(df.head())
# print(df.shape)
# print(df.info())

# data cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns='Cabin',inplace=True)


df['Sex']=df['Sex'].map({'male':0,'female':1})
df['Embarked']=df['Embarked'].map({'S':0,'C':1,'Q':2})

print(df[['Age','Embarked','Sex']].head(10))

# check null
print(df.isnull().sum())


# separate feature x from y

X=df[['Pclass','Sex','Age','Embarked']]
Y=df['Survived']

# splitting into train and test 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

predictions = model.predict(x_test)

print("accuracy",accuracy_score(y_test,predictions))

model2=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
predictions2=model.predict(x_test)


print("randomclassifier score",accuracy_score(y_test,predictions2))


# Feature Engineering — naye features banao
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Updated features list
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model3 = RandomForestClassifier(n_estimators=200, random_state=42)
model3.fit(X_train, y_train)

predictions3 = model3.predict(X_test)
print("Improved Random Forest:", accuracy_score(y_test, predictions3))





# Best settings dhundho automatically
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

print("Best Settings:", grid.best_params_)
print("Best Accuracy:", accuracy_score(y_test, grid.predict(X_test)))


from sklearn.metrics import classification_report, confusion_matrix

best_predictions = model3.predict(X_test)

print("Final Accuracy:", accuracy_score(y_test, best_predictions))
print("\nDetailed Report:")
print(classification_report(y_test, best_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_predictions))

import joblib

# Model save karo
joblib.dump(model3, 'titanic_model.pkl')
print("Model saved!")

# Test karo ke save hua
loaded_model = joblib.load('titanic_model.pkl')
print("Loaded accuracy:", accuracy_score(y_test, loaded_model.predict(X_test)))