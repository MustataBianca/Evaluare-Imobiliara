import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from implementare import data, b, test, b_test

def scale_data(train_df, test_df):
    # (x - medie) / deviatie_standard  scaleaza criteriile pentru a avea un impact proportional
    means = train_df.mean()
    stds = train_df.std()
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
            
            # actualizarea coloanei k in A
            A[k, k] = -sigma
            A[k+1:m, k] = 0
            
            # aplicarea transformarii pe restul coloanelor
            for j in range(k + 1, n):
                tau = np.dot(U[k:m, k], A[k:m, j]) / beta[k]
                A[k:m, j] = A[k:m, j] - tau * U[k:m, k]
                    
    return A, U, beta

def cmmp(A, b_vec):
    m, n = A.shape
    R_full, U, beta = tort(A)
    
    # calculul vectorului d care suprascrie b
    d = b_vec.copy().astype(float)
    for k in range(n):
        if beta[k] != 0:
            tau = np.dot(U[k:m, k], d[k:m]) / beta[k]
            d[k:m] = d[k:m] - tau * U[k:m, k]
    
    # UTRIS
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
A_final = np.hstack((np.ones((A_scaled.shape[0], 1)), A_scaled))
T_final = np.hstack((np.ones((T_scaled.shape[0], 1)), T_scaled))

# CMMP manual
coef_manual = cmmp(A_final, b)
pred_manual = T_final @ coef_manual

# numpy.linalg.lstsq care returneaza solutie, reziduuri, rang, valori singulare
coef_standard, _, _, _ = np.linalg.lstsq(A_final, b, rcond=None)
pred_standard = T_final @ coef_standard

rezultate_complet = pd.DataFrame({
    'ID': np.arange(len(b_test)) + 1,
    'Pret Real': b_test,
    'Estimat Manual': pred_manual,
    'Estimat lstsq': pred_standard,
    'Dif. Manual vs lstsq': pred_manual - pred_standard
})

pd.options.display.float_format = '{:,.2f}'.format
print("Comparatie CMMP: manual vs. lstsq")
print(rezultate_complet.to_string(index=False))

eroare_metode = np.linalg.norm(coef_manual - coef_standard)
print(f"\nDiferenta de eroare dintre cele doua metode: {eroare_metode:.2e}")

def plot_scatter_comparison(y_real, y_pred):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_real, y_pred, color='blue', alpha=0.6, label='Apartamente Test')
    
    limite = [min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())]
    plt.plot(limite, limite, color='red', linestyle='--', linewidth=2, label='Referință (Ideal)')
    
    plt.title('Comparație: Preț Real vs. Preț Estimat', fontsize=14)
    plt.xlabel('Preț Real (EURO)', fontsize=12)
    plt.ylabel('Preț Estimat (EURO)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    plt.show()

plot_scatter_comparison(b_test, pred_manual)