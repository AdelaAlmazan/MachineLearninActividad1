import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler



categorias_encoder = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
categorias_model= ['Linear Regression','Decision Tree', 'RandomForestRegressor']
cat_encoder = OneHotEncoder(categories=categorias_encoder)

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


standar = StandardScaler()

standar.mean_ = [-1.19575635e+02,  3.56393144e+01,  2.86534036e+01,  2.62253979e+03,
         5.33939438e+02,  1.41968738e+03,  4.97011810e+02,  3.87588428e+00,
         5.44040595e+00,  3.09646921e+00,  2.13697971e-01]

standar.scale_ = [2.00176745e+00, 2.13789807e+00, 1.25744378e+01, 2.13835233e+03,
        4.10793820e+02, 1.11562925e+03, 3.75684780e+02, 1.90487283e+00,
        2.61161689e+00, 1.15844737e+01, 6.53420203e-02]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

imputer.statistics_ = [-118.51   ,   34.26   ,   29.     , 2119.     ,  433.     , 1164.     ,  408.     ,    3.54155]

num_pipeline=Pipeline([
    ("imputer", imputer),
    ("attrins_adder", CombinedAttributesAdder()),
    ("std_scaler", standar),
])


from sklearn.compose import ColumnTransformer

housing_num = ['longitude',
 'latitude',
 'housing_median_age',
 'total_rooms',
 'total_bedrooms',
 'population',
 'households',
 'median_income']

num_attribs= list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline,num_attribs),
    ("cat", cat_encoder,cat_attribs),
])



st.title('Modelo de Machine Learning end to end', anchor='big-font')



# Cargar el modelo
model = joblib.load('C:\\Users\\adela\\OneDrive\\Escritorio\\EndToEndML\\bestmodelHousing.pkl')
model_linear = joblib.load('C:\\Users\\adela\\OneDrive\\Escritorio\\EndToEndML\\LinearRegression (2).pkl')
modelo_tree_reg = joblib.load('C:\\Users\\adela\\OneDrive\\Escritorio\\EndToEndML\\modelo_tree_reg (3).pkl')

# Crear una interfaz de usuario con Streamlit
st.title('Predicción del Valor de la Vivienda', anchor='small-font')

with st.sidebar.expander("Descripción de la Propiedad"):
####INPUT
    longitude = st.number_input('longitude', value=1.0)
    latitude = st.number_input('latitude',  value=1.0)
    housing_median_age = st.number_input('housing_median_age',  value=1.0)
    total_rooms=  st.number_input('total rooms', value=1.0)
    total_bedrooms=  st.number_input('total bedrooms', value=1.0)
    population=  st.number_input('population', value=1.0)
    median_income = st.number_input('Media income', value=1.0)
    households=  st.number_input('households', value=1.0)
    ocean_proximity = st.selectbox('Selecciona una opción de ocean_proximity', categorias_encoder)

with st.sidebar.expander("Tipo de modelo"):

    type_model = st.selectbox('Seleciona con que modelo predicir', categorias_model)



# Crear un botón para hacer predicciones
if st.button('Predecir'):
    # Crear un diccionario con los datos ingresados por el usuario
    input_data = {
 
'longitude' :  longitude,
'latitude':latitude  ,
'housing_median_age' : housing_median_age,
'total_rooms': total_rooms,
'total_bedrooms': total_bedrooms,
'population':population ,
'households':households ,
'median_income': median_income ,
'ocean_proximity':ocean_proximity 
    }
    
    with open('C:\\Users\\adela\\OneDrive\\Escritorio\\EndToEndML\\full_pipeline(2).pkl', 'rb') as f:
        full_pipeline = pickle.load(f) 

    input_data_df = pd.DataFrame([input_data])
    
    input_adjusted = full_pipeline.transform(input_data_df)


    if type_model == 'Linear Regression':
        prediction = model_linear.predict(input_adjusted)
        st.markdown(f'**Modelo Utilizado:** {type_model}')
        st.success(f'**Predicción del Valor de la Vivienda:** ${prediction[0]:,.2f}')
    
    elif type_model == 'Decision Tree':
        prediction = modelo_tree_reg.predict(input_adjusted)
        st.markdown(f'**Modelo Utilizado:** {type_model}')
        st.success(f'**Predicción del Valor de la Vivienda:** ${prediction[0]:,.2f}')

    elif type_model == 'RandomForestRegressor':
        prediction = model.predict(input_adjusted)
        st.markdown(f'**Modelo Utilizado:** {type_model}')
        st.success(f'**Predicción del Valor de la Vivienda:** ${prediction[0]:,.2f}')


    