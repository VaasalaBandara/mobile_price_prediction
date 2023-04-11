# -*- coding: utf-8 -*-


import pandas as pd
import tensorflow as tf


df=pd.read_csv("mobile_price_classification.csv")

df.head(5)

df.describe()

X=df.drop(columns=['price_range'])

X

y=df['price_range']

y

print(df.isnull().any()) #to determine whether there are any null values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.75)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(units=10,activation='tanh'),
    tf.keras.layers.Dense(units=8,activation='relu'),
    tf.keras.layers.Dense(units=4,activation='relu'),
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])



model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100)

model.save('weights.h5')

