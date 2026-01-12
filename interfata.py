import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from implementare import data, b, test, b_test


# Re-executarea logicii de pre-procesare din main.py
AN_CURENT = 2025
df_train = pd.DataFrame(data)
df_train['Varsta'] = AN_CURENT - df_train['An_construire']
df_train.drop('An_construire', axis=1, inplace=True)
COLOANE_MODEL = df_train.columns.tolist() 

# Scalare și Antrenare Model Ridge (CMMP Stabilizat)
scaler = StandardScaler()
A_scaled = scaler.fit_transform(df_train)
intercept_A = np.ones((A_scaled.shape[0], 1))
A_final = np.hstack((intercept_A, A_scaled))
model = Ridge(alpha=100.0) 
model.fit(A_final, b) 

def predict_price(input_data):
    input_data = input_data.copy()
    
    input_data['Varsta'] = AN_CURENT - input_data['An_construire']
    input_data.drop('An_construire', axis=1, inplace=True)
    
    input_df = input_data[COLOANE_MODEL] 
    X_scaled = scaler.transform(input_df)
    
    # Adaugă Interceptul și face predicția
    X_final = np.hstack((np.ones((1, 1)), X_scaled))
    pret_estimat = model.predict(X_final)[0]
    
    return pret_estimat

# Interfata
st.set_page_config(layout="centered")
st.title("Estimator de Preț Apartament (Model CMMP Stabilizat)")
st.markdown("Introduceți detaliile apartamentului dorit pentru a obține un preț estimativ.")

# Crearea câmpurilor de intrare
with st.form("predictie_form"):
    st.header("Detalii Apartament")
    
    col1, col2 = st.columns(2)
    suprafata = col1.number_input("Suprafață Utila (mp)", min_value=30.0, max_value=200.0, value=65.0, step=0.5)
    nr_camere = col2.selectbox("Număr Camere", [1, 2, 3, 4, 5], index=2)
    
    col3, col4 = st.columns(2)
    distanta_centru = col3.number_input("Distanță Centru (km)", min_value=0.5, max_value=20.0, value=5.0, step=0.1)
    an_construire = col4.slider("An Construire", min_value=1950, max_value=2025, value=2000)
    
    col5, col6 = st.columns(2)
    etaj = col5.number_input("Etaj (0=Parter)", min_value=0, max_value=10, value=3)
    nr_bai = col6.selectbox("Număr Băi", [1, 2, 3, 4], index=0)
    
    stare_generala = st.select_slider("Stare Generală (1=Slabă, 5=Excelentă)", options=[1, 2, 3, 4, 5], value=4)
    mobilat = st.radio("Mobilat?", options=[1, 0], format_func=lambda x: "Da" if x == 1 else "Nu")
    
    submitted = st.form_submit_button("Estimează Prețul")

# Logica de predicție
if submitted:
    input_data = pd.DataFrame({
        'Suprafata_mp': [suprafata],
        'Nr_Camere': [nr_camere],
        'Distanta_Centru_km': [distanta_centru],
        'mobilat': [mobilat],
        'etaj': [etaj],
        'nr_bai': [nr_bai],
        'stare_generala': [stare_generala],
        'An_construire': [an_construire], 
    })
    
    try:
        pret_estimat = predict_price(input_data)
        
        st.success("Predicție finalizată!")
        st.markdown(f"### Prețul Estimat al Apartamentului este:")
        st.markdown(f"**{pret_estimat:,.0f} EURO**")

        st.info("Rețineți: Această estimare este bazată pe un model antrenat pe doar 50 proprietăți. Rezultatele sunt stabile, dar marja de eroare poate fi mare.")
        
    except Exception as e:
        st.error(f"Eroare la estimare: {e}")

        #python -m streamlit run interfata.py