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
    #print(df.columns)
    return df

def pie_chart_per_year_per_standard_index(df):
    # Create a 3x3 (3 years × 3 standards)  grid of subplots (piecharts)
    pieplots, axes = plt.subplots(3, 3, figsize=(13, 9))
    # where pieplots is the name of the figure
    #   and axes creates a 3x3 ixj empty array to the later fill
    pieplots.suptitle('INSERIRE TITOLO ADEGUATO (2022-2024)', fontsize=16, fontweight='bold')

    # Creating the columns of the standards per year more effeciently
    standards = ['GRI', 'ESRS', 'SASB']
    years = [2022, 2023, 2024]

    # Iterate through years (rows) and standards (columns)
    for i, standard in enumerate(standards):
        for j, year in enumerate(years):
            # Get the data for this year and standard
            df_column_name = f'{standard}_{year}'
            value_counts = df[df_column_name].value_counts(normalize=True)

            # Create pie chart in the appropriate subplot, where
            #   axes[i, j] lets you target specific subplot positions
            axes[i, j].pie(value_counts.values,
                           labels=['Non adottato', 'Adottato'],
                           autopct='%1.1f%%',
                           startangle=90,
                           colors=['lightcoral', 'lightgreen'])
            axes[i, j].set_title(f'{standard} {year}', fontweight='bold')

    plt.tight_layout()  # Prevents overlapping
    plt.savefig('INSERIRE TITOLO ADEGUATO (2022-2024).png')
    #plt.show()

path = "Database ufficiale.csv"
df1 = pd.read_csv(path)

#Reshaping the database with GRI, ESRS, SASB and widening the columns of the previous indexes into the three years
    #and creating a new cleaner csv
df = reshape_database(df1)

df.to_csv('Tidier_Dataset.csv', index=False)
df.to_excel('Tidier_Dataset.xlsx', index=False)

#Creating a table of piechart with the Usage or not of each index
pie_chart_per_year_per_standard_index(df)