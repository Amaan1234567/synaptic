from engine import MLP,SGD,Utils,Value
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random as r
import math

scaler = MinMaxScaler()

X,y=load_breast_cancer(return_X_y=True)
print(X.shape)
print(y[123])
X = scaler.fit_transform(X)
new_X=[]
new_y=[]
for data in X:
    new_data = [Value(i) for i in data]
    new_X.append(new_data)

print(y)
for data in y:
    new_y.append(Value(data))

X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.333, random_state=42)
#print(X_test.shape)



model = MLP(nin=30,nouts=[15,1])

iters=1000
lr=0.001

for it in range(iters):
    idx=r.randint(0,len(X_train)-1)
    x,y=X_train[idx],y_train[idx]
    #print(y)
    y_preds=model(x)
    y_preds = Utils.sigmoid(y_preds)
    print("predictions: ",y_preds)
    loss = (y_preds-y)**2
    #lr *=0.999
    print("performing backward")
    loss.backward()
    SGD.optmiser_step(model,lr)
    print(f"{it} iteration, loss={loss.data}")

final_acc=0
for x,y in zip(X_test,y_test):
    y_preds=model(x)
    if((y_preds.data>0.5 and y.data==1) or (y_preds.data<0.5 and y.data==0)):
        final_acc+=1
print(final_acc)
print(f"acc on test data: {(final_acc/len(X_test))*100}")