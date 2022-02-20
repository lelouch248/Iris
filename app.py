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
a = st.slider(label = 'Sepal Length',min_value = 0.50,max_value = 7.90,value = 5.1)
b = st.slider(label = 'Sepal Width',min_value = 0.50,max_value = 4.40,value = 3.5)
c = st.slider(label = 'Petal Length',min_value = 0.50,max_value = 6.90,value = 1.4)
d = st.slider(label = 'Petal Width',min_value = 0.10,max_value = 6.90,value = 0.2)
result = model.predict([[a,b,c,d]])
st.title(iris.target_names[result])

  
