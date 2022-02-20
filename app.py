#%%writefile app.py
import streamlit as st
import joblib
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

df = pd.DataFrame(iris.data)
df['Target'] = iris.target

from pandas.core.common import not_none
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)

st.title('IRIS CLASSIFIER - API')
a = st.slider(label = 'Sepal Length',min_value = 0.50,max_value = 7.90)
b = st.slider(label = 'Sepal Width',min_value = 0.50,max_value = 4.40)
c = st.slider(label = 'Petal Length',min_value = 0.50,max_value = 6.90)
d = st.slider(label = 'Petal Width',min_value = 0.50,max_value = 6.90)
result = model.predict([[a,b,c,d]])
if result == 0:
  st.text('Setosa')
if result == 1:
  st.text('versicolor')
if result ==2:
  st.text('verginica')

  
