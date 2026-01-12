import numpy as np
import pandas as pd
from implementare import data, b, test, b_test

def scale_data(train_df, test_df):
    # calculeaza media si deviatia standard pentru fiecare coloana.
    means = train_df.mean()
    stds = train_df.std()
    
    # (x - medie) / deviatie_standard
    train_scaled = (train_df - means) / stds
    test_scaled = (test_df - means) / stds
    
    return train_scaled.to_numpy(), test_scaled.to_numpy()

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

            # actualizarea coloanei k în A
            A[k, k] = -sigma
            A[k+1:m, k] = 0 

            # actualizarea coloanelor j = k+1:n 
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

df_train = pd.DataFrame(data)
df_test = pd.DataFrame(test)

AN_CURENT = 2026
for df in [df_train, df_test]:
    df['Varsta'] = AN_CURENT - df['An_construire']
    df.drop('An_construire', axis=1, inplace=True)

A_scaled, T_scaled = scale_data(df_train, df_test)

# adaugare coloana de 1 pentru constanta modelului
A_final = np.hstack((np.ones((A_scaled.shape[0], 1)), A_scaled))
T_final = np.hstack((np.ones((T_scaled.shape[0], 1)), T_scaled))
coeficienti = cmmp(A_final, b)

preturi_estimate = T_final @ coeficienti

eroare_absoluta = np.abs(preturi_estimate - b_test)
eroare_procentuala = (eroare_absoluta / b_test) * 100

rezultate_df = pd.DataFrame({
    'ID Test': np.arange(len(b_test)) + 1,
    'Pret Real': b_test,
    'Pret Estimat': preturi_estimate,
    'Eroare Absoluta': eroare_absoluta,
    'Eroare %': eroare_procentuala
})

pd.options.display.float_format = '{:,.2f}%'.format
print(rezultate_df.to_string(index=False))