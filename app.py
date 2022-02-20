%%writefile app.py
import streamlit as st
import joblib
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
