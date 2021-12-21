# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 00:29:39 2021

@author: ADMIN
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
#df = pd.read_excel(r'Data_Train.xlsx')
#print(df)
dataset = pd.read_csv('addresses.csv')
print(dataset)
dataset['experience'].fillna(0, inplace=True)

#print(dataset)
dataset['interview_score'].fillna(0, inplace=True)
#mean1=dataset['test_score']
#print(mean1)
#dataset.iloc[:,1].fillna(0, inplace=True)

dataset['interview_score'].fillna((dataset['interview_score'].mean()), inplace=True)
dataset.iloc[:,1].fillna((dataset.iloc[:,1].mean()), inplace=True)

X = dataset.iloc[:, :3]
print(X)
y=dataset.iloc[:,3]
print(y)
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))
print(X)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))




