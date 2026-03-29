from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np

df =pd.read_csv("trained.csv")
print(df.head())

df['Sex'] =df['Sex'].map({"male":0,"female":1})

df["Age"] =df["Age"].fillna(df["Age"].median())

df.drop(columns=['Cabin'],inplace =True)
print(df)
df['FamilySize'] = df['SibSp'] + df['Parch']
X =df[['Pclass', 'Sex', 'Age','FamilySize', 'Fare']]
y=df['Survived']

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

clf =DecisionTreeClassifier(max_depth=3)
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
print("Accuracy :-",accuracy_score(y_test,y_pred))
print(y_pred)

print("======New Survival Predictor ======")
pclass =int(input("Passanger Class (1|2|3)-"))
sex =input("Sex (Male|Female)-")
age =int(input("Enter the age -"))
sibsp =int(input("Enter the (Sibling-0|Spouse-1):-"))
parch =int(input("Parents-0|Children-1-"))
fare = float(input("Fare:-"))

sex =1 if sex =="Female" else 0
familySize =sibsp +parch

new_pas =[[pclass,sex,age,familySize,fare]]
prediction =clf.predict(new_pas)


if prediction[0] ==1:
    print("Survived")
else :
    print("Not Survived!!")