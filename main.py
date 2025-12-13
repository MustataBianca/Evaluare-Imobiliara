import numpy as np
import pandas as pd
from implementare import data, b

m = data.shape[0] # nr proprietati

intercept = np.ones((m, 1))
A = np.stack((intercept, data)) # poate de adaugat in implementare

coef, reziduu, rang = np.linalg.lstsq(A, b, rcond=None) #CMMP Ax=b

#de introdus exemple

