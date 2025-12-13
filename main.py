import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from numpy.linalg import lstsq
from implementare import data, b, test, b_test

df_train = pd.DataFrame(data)
df_test = pd.DataFrame(test)

AN_CURENT = 2025
for df_temp in [df_train, df_test]:
    df_temp['Varsta'] = AN_CURENT - df_temp['An_construire']
    df_temp.drop('An_construire', axis=1, inplace=True)

scaler = StandardScaler()

A_scaled = scaler.fit_transform(df_train)
T_scaled = scaler.transform(df_test)

intercept_A = np.ones((A_scaled.shape[0], 1))
intercept_T = np.ones((T_scaled.shape[0], 1))

A_final = np.hstack((intercept_A, A_scaled))
T_final = np.hstack((intercept_T, T_scaled))

model = Ridge(alpha=100.0) 
model.fit(A_final, b) 

preturi_estimate = model.predict(T_final)
eroare_absoluta = np.abs(preturi_estimate - b_test)
eroare_procentuala = (eroare_absoluta / b_test) * 100

# Construim DataFrame-ul de comparație
rezultate_df = pd.DataFrame({
    'ID Test': np.arange(len(b_test)) + 1,
    'Pret Real (EURO)': b_test,
    'Pret Estimat (EURO)': preturi_estimate,
    'Eroare Absoluta (EURO)': eroare_absoluta,
    'Eroare Procentuala (%)': eroare_procentuala
})

pd.options.display.float_format = '{:,.0f}'.format 
rezultate_df['Eroare Procentuala (%)'] = rezultate_df['Eroare Procentuala (%)'].map('{:,.2f}%'.format)
print("Model CMMP Stabilizat (Ridge + StandardScaler) - Fără Prețuri Negative")
print(f"Număr de rânduri în antrenare: {A_final.shape[0]}")
print(f"Număr de coeficienți calculați: {A_final.shape[1]}")
print("\nComparație Preț Real vs. Estimat:")
print(rezultate_df.to_markdown(index=False, numalign="left", stralign="left"))