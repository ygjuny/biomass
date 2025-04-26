# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:26:12 2024

@author: lenovo
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd 



def func3(x,a,b,c):
    return a + b *x + c * x*x

data1 = pd.read_excel(r'***.xlsx',sheet_name='Sheet1')

x1 = np.array(data1.iloc[:,1])
ys = np.array(data1.iloc[:,2:])

abc = np.zeros([36,3])
for i in range(ys.shape[1]):
    y = ys[:,i]
    popt, pcov = curve_fit(func3, x1, y)
    a_fit, b_fit, c_fit = popt
    print("FIT：")
    print("a =", a_fit)
    print("b =", b_fit)
    print("c =", c_fit)
    abc[i,0],abc[i,1],abc[i,2] = a_fit, b_fit, c_fit
