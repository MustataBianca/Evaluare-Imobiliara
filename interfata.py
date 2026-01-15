import streamlit as st
import numpy as np
import pandas as pd
from implementare import data, b, test, b_test


def tort(A_in):
    A = A_in.copy().astype(float)
    m, n = A.shape
    p = min(m - 1, n) 
    U = np.zeros((m, n)) 
    beta = np.zeros(n)

    for k in range(p): 
        sigma = np.sign(A[k, k]) * np.sqrt(np.sum(A[k:m, k]**2))
        if sigma == 0:
            beta[k] = 0 
        else:
            U[k, k] = A[k, k] + sigma
            U[k+1:m, k] = A[k+1:m, k]
            beta[k] = sigma * U[k, k]
            A[k, k] = -sigma
            A[k+1:m, k] = 0 

            for j in range(k + 1, n):
                tau = np.dot(U[k:m, k], A[k:m, j]) / beta[k]
                A[k:m, j] = A[k:m, j] - tau * U[k:m, k]
    return A, U, beta

def cmmp(A, b_vec):
    m, n = A.shape
    R_full, U, beta = tort(A) 
    d = b_vec.copy().astype(float)
    for k in range(n): 
        if beta[k] != 0:
            tau = np.dot(U[k:m, k], d[k:m]) / beta[k] 
            d[k:m] = d[k:m] - tau * U[k:m, k] 
    
    R_prime = R_full[:n, :n]
    d_prime = d[:n]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = np.dot(R_prime[i, i + 1:], x[i + 1:])
        x[i] = (d_prime[i] - suma) / R_prime[i, i]
    return x



AN_CURENT = 2026
df_train = pd.DataFrame(data)
df_train['Varsta'] = AN_CURENT - df_train['An_construire']
df_train.drop('An_construire', axis=1, inplace=True)


means = df_train.mean()
stds = df_train.std()

def scale_custom(df):
    return (df - means) / stds


A_scaled = scale_custom(df_train).to_numpy()
A_final = np.hstack((np.ones((A_scaled.shape[0], 1)), A_scaled))
coeficienti = cmmp(A_final, b)



st.set_page_config(layout="centered", page_title="Estimator Apartamente")
st.title("Estimator de Preț (Algoritm Householder)")
st.markdown("Acest model folosește implementarea manuală a metodei Celor Mai Mici Pătrate.")

with st.form("predictie_form"):
    st.header("Detalii Apartament")
    col1, col2 = st.columns(2)
    suprafata = col1.number_input("Suprafață Utila (mp)", min_value=30.0, max_value=200.0, value=65.0)
    nr_camere = col2.selectbox("Număr Camere", [1, 2, 3, 4, 5], index=2)
    
    col3, col4 = st.columns(2)
    distanta_centru = col3.number_input("Distanță Centru (km)", min_value=0.5, max_value=20.0, value=5.0)
    an_construire = col4.slider("An Construire", 1920, 2026, 2000)
    
    col5, col6 = st.columns(2)
    etaj = col5.number_input("Etaj", 0, 15, 3)
    nr_bai = col6.selectbox("Număr Băi", [1, 2, 3], index=0)
    
    stare_generala = st.select_slider("Stare Generală", options=[1, 2, 3, 4, 5], value=4)
    mobilat = st.radio("Mobilat?", options=[1, 0], format_func=lambda x: "Da" if x == 1 else "Nu")
    
    submitted = st.form_submit_button("Calculează Prețul")

if submitted:
    # creare de dataframe cu inputul
    input_df = pd.DataFrame({
        'Suprafata_mp': [suprafata],
        'Nr_Camere': [nr_camere],
        'Distanta_Centru_km': [distanta_centru],
        'mobilat': [mobilat],
        'etaj': [etaj],
        'nr_bai': [nr_bai],
        'stare_generala': [stare_generala],
        'Varsta': [AN_CURENT - an_construire]
    })
    
    #  scalare folosind mediile de la antrenare
    X_scaled = scale_custom(input_df).to_numpy()
    X_final = np.hstack((np.ones((1, 1)), X_scaled))
    
    #  predicția reprezinta produsul scalar intre input si coeficientii calculati de cmmp
    pret_estimat = (X_final @ coeficienti)[0]
    
    st.success(f"Preț Estimat: {pret_estimat:,.0f} EURO")
    st.info("Rețineți: Această estimare este bazată pe un model antrenat pe doar 100 proprietăți. Rezultatele sunt stabile, dar marja de eroare poate fi mare.")