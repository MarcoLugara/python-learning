import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
#matplotlib inline
import matplotlib.pyplot as plt

def reshape_database(df1):
    """
    Complete database reshaping including column renaming and pivot transformation
    """
    #Rename columns for readability
    df2 = df1.rename(columns={'AZIENDE SUDDIVISE PER CODICE ATECO': 'Name', 'ATECO 2007\ncodice': 'ATECO', 'ATECO 2007\ndescrizione': 'ATECOx'})

    #Reshape the database to just have the 3 Boolean variables to analyze, namely GRI, ESRS, SASB
    df3= df2[['Name', 'Anno', 'ATECO', 'ATECOx', 'GRI', 'ESRS', 'SASB']]

    # Create a wide format with years as columns
    df = df3.pivot_table(
        index=['Name', 'ATECO', 'ATECOx'],  # Keep these as rows
        columns='Anno',           # Years become columns
        values=['GRI', 'ESRS', 'SASB']  # These become values per year
    ).reset_index()
    #print(df.columns) this would, for now, print them as tuples of variable and
    # year to recognize where is what, so it's best to look at it and rename it to make it more clean

    # Flatten the column names
    df.columns = ['Name', 'ATECO', 'ATECOx', 'GRI_2022', 'GRI_2023', 'GRI_2024',
                       'ESRS_2022', 'ESRS_2023', 'ESRS_2024',
                       'SASB_2022', 'SASB_2023', 'SASB_2024']

    """
    print(f"Original shape: {df3.shape}")  --> Original shape: (876, 7)
    print(f"Wide shape: {df.shape}") --> Wide shape: (291, 12) 291 companies analyzed

     Checks:
     name_list = df['Name'].tolist()
     print(f"Total entries: {len(name_list)}") --> Total entries: 291
    """
    print(df.columns)
    return df

path = "DATASET.csv"
df1 = pd.read_csv(path)

#Reshaping the database with GRI, ESRS, SASB and widening the columns of the previous indexes into the three years
df = reshape_database(df1)