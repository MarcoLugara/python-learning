import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
#matplotlib inline
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)

path = "NHANES.csv"
df = pd.read_csv(path)

unique_values = df["SEQN"].unique()
print(unique_values)
