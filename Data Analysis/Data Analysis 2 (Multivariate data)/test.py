import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
#matplotlib inline
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)

path = "NHANES.csv"
df = pd.read_csv(path)
bp = df['BPXSY2']
bp_median = bp.median()
print(bp_median)

bp_mean = bp.mean()
print(bp_mean)

bp_sd = bp.std()
print(bp_sd)

bp_max = bp.max()
print(bp_max)

bp_IQR = bp.quantile(0.75) - bp.quantile(0.25)
print(bp_IQR)
