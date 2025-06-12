import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.linalg import lu_factor, lu_solve
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from tifffile import TiffFile
from pathlib import Path
from scipy.optimize import curve_fit
import datetime
import csv
import pandas as pd
import math

#*水温が室温以上の領域の温度分布の近似
def approximation_temp_cutoff(dic,T_room):
    def exp_fit(x,a,b,c):
        y = a*np.exp(-c*(x-b)**2) + T_room #*各パラメーターが１桁になるようにオーダーを調整
        return y
    array_x = np.array(list(dic.keys()))
    array_y = np.array(list(dic.values()))
    popt, pcov = curve_fit(exp_fit ,array_x, array_y, p0 = [20,0,0.1])
    list_y = []
    for num in dic.keys():
        list_y.append(popt[0]*np.exp(-popt[2]*(num - popt[1])**2) + T_room)
    dic2 = dict(zip(list(dic.keys()), list_y))
    myList = dic2.items()
    myList = sorted(myList)
    x, y = zip(*myList)
    return x, y, popt

#*水温が室温以上の領域の流速分布の近似
def approximation_flow_cutoff(dic):
    def exp_fit(x,a,b,c,d,e):
        y = a*np.exp(-c*(x-b)**2) + d*x + e #*各パラメーターが１桁になるようにオーダーを調整
        return y
    array_x = np.array(list(dic.keys()))
    array_y = np.array(list(dic.values()))
    popt, pcov = curve_fit(exp_fit ,array_x, array_y, p0 = [0.03,0,0.1,0.01,0.01])
    list_y = []
    for num in dic.keys():
        list_y.append(popt[0]*np.exp(-popt[2]*(num - popt[1])**2) + popt[3]*num + popt[4])
    dic2 = dict(zip(list(dic.keys()), list_y))
    myList = dic2.items()
    myList = sorted(myList)
    x, y = zip(*myList)
    return x, y, popt
