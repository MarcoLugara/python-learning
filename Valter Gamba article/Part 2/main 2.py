import numpy as np
import pandas as pd
import seaborn as sns
#import scipy.stats as stats
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
#from matplotlib.ticker import PercentFormatter
import csv
import re


#OUTDATED/EXECUTE PROCESSES
def reshape_database(df1):
    """
    Complete database reshaping using a manual approach
    """
    # Rename columns for readability
    df2 = df1.rename(columns={
        'AZIENDE SUDDIVISE PER CODICE ATECO': 'Name',
        'ATECO 2007\r\ncodice': 'ATECO',
        'ATECO 2007\r\ndescrizione': 'ATECOx',
        'APPLICAZIONE STANDARD AA1000': 'AA1000'}
    )

    # Get unique companies
    companies = df2['Name'].unique()

    # Create empty result DataFrame
    result_data = []

    for company in companies:
        company_data = df2[df2['Name'] == company]

        # Get base info
        base_info = company_data.iloc[0][['Name', 'ATECO', 'ATECOx']]
        row_data = {
            'Name': base_info['Name'],
            'ATECO': base_info['ATECO'],
            'ATECOx': base_info['ATECOx']
        }

        # Add AA1000 for each year
        for year in [2022, 2023, 2024]:
            year_data = company_data[company_data['Anno'] == year]
            if not year_data.empty:
                row_data[f'AA1000_{year}'] = year_data['AA1000'].iloc[0]
            else:
                row_data[f'AA1000_{year}'] = 0

        result_data.append(row_data)

    df = pd.DataFrame(result_data)

    # Handle missing values and data types
    df = df.replace('#N/A', np.nan)

    # Convert all indicator columns to integers
    indicator_cols = [col for col in df.columns if col.startswith(('AA1000'))]
    for col in indicator_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Convert ATECO to appropriate type if needed
    df['ATECO'] = pd.to_numeric(df['ATECO'], errors='coerce')

    # Delete the last useless row
    df = df.drop(df.index[-1])

    # Print in used formats
    df.to_csv('Tidier_Dataset.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)

    #ADDING THE EXTRA ATECO NOMINATIONS: PART 1
    df_Extra = pd.read_csv('../Part 2/ATECO_codes.csv')
    df_Extra = df_Extra.rename(columns={
        'Codice Ateco': 'Codice',
        'Titolo Ateco 2007 aggiornamento 2022': 'Codice_desc'}
    )
    df1 = df_Extra[['Codice', 'Codice_desc']]
    df1.to_csv('step1.csv', index=False)
    df1.to_excel('step1.xlsx', index=False)

    input_file = "step1.csv"
    output_file = "step2.csv"

    rows = []

    # Read the CSV file - FIXED INDENTATION
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Moved this inside the with block
        for row in reader:
            codice = row['Codice'].strip()
            # Check if Codice is NOT a single letter (A-Z)
            if not re.match(r'^[A-Z]$', codice):
                row['Codice_desc'] = ''
            rows.append(row)
        # print(rows)

    # Write the modified data
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"File processed successfully. Output saved to: {output_file}")


    return df

#ACTUAL CODE
path1 = "Database Ufficiale.csv"       #Old naming for DF processing
df1= pd.read_csv(path1)
print(df1.columns)
reshape_database(df1)

