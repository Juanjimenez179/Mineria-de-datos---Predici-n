# -*- coding: utf-8 -*-
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Cargar el modelo
filename = 'modelo-reg.pkl'
model_Tree, model_Knn, model_NN, min_max_scaler, variables = pickle.load(open(filename, 'rb'))

# Interfaz gráfica
st.title('Predicción del valor de arriendo de un inmueble')

tipo_inmueble = st.selectbox('Tipo de Inmueble', ['Apartamento', 'Casa', 'Estudio'])
departamento = st.selectbox('Departamento', ['Bogotá D.C.', 'Valle del Cauca'])
estrato = st.slider('Estrato', min_value=1, max_value=6, value=3)

# Crear DataFrame
datos = pd.DataFrame([[tipo_inmueble, departamento, estrato]],
                     columns=['Tipo Inmueble', 'Departamento', 'Estrato'])

# One-hot encoding
datos_encoded = pd.get_dummies(datos, columns=['Tipo Inmueble', 'Departamento'], drop_first=False)

# Reindexar para que coincida con las columnas del modelo
datos_encoded = datos_encoded.reindex(columns=variables, fill_value=0)

# Predicción con el modelo de árbol
prediccion = model_Tree.predict(datos_encoded)

# Mostrar resultado
st.subheader(f'Valor estimado de arriendo: ${int(prediccion[0]):,}')

