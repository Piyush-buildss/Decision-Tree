from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 
import pandas as pd 

df =pd.read_csv("heart.csv")

X =df[['cp','trestbps','thalach','oldpeak','ca','thal']]
y =df['target']

X_train ,X_test ,y_train ,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

clf =DecisionTreeClassifier(max_depth=3)
clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
print("Accuracy -",accuracy_score(y_test,y_pred))


print("======Predictor Disase ======")
cp = int(input("Enter the chest pain type (0|1|2|3)-"))
trestbps =int(input("Enter the trestbps -"))
thalach =int(input("Enter the thalach -"))
oldpeak =float(input('Oldpeak -'))
ca = int(input("Enter Ca (0|1|2 |3)-"))
thal = int(input("Thal(0|1|2|3) -"))

new_X =[[cp,trestbps,thalach,oldpeak,ca,thal]]
prediction = clf.predict(new_X)

if prediction[0]  ==0:
    print("You not  have heart disease")
else :
    print("You have heart disease")