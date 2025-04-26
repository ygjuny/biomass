# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:24:37 2024

@author: lenovo
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd 

def func2(x, a, b, c,d,k):
    return k + a * x[:,0] + b * x[:,1] + c * x[:,2] + d * x[:,3]

data1 = pd.read_excel(r'***.xlsx',sheet_name='PV')
data2 = pd.read_excel(r'***.xlsx',sheet_name='VI')

abc = np.array(data1.iloc[:,1:])
x2 = np.array(data2.iloc[:,1:])

ks = np.zeros([3,5])
for j in range(3):
    popt, pcov = curve_fit(func2, x2, abc[:,j])
    a_fit1, b_fit1, c_fit1, d_fit1, k_fit1 = popt
    print("FIT：")
    print("a =", a_fit1)
    print("b =", b_fit1)
    print("c =", c_fit1)
    print("d =", d_fit1)
    print("k =", k_fit1)
    ks[j,0],ks[j,1],ks[j,2],ks[j,3],ks[j,4] = a_fit1, b_fit1, c_fit1, d_fit1, k_fit1