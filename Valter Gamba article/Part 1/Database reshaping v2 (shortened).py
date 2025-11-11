import numpy as np
import pandas as pd
import re
import csv


def reshape_complete_database(main_db_path, ateco_codes_path):
    """
    Complete database reshaping function that handles all transformation steps
    from raw data to final formatted dataset with ATECO classifications.

    Parameters:
    -----------
    main_db_path : str
        Path to the main database CSV file
    ateco_codes_path : str
        Path to the ATECO codes CSV file

    Returns:
    --------
    pd.DataFrame
        Fully reshaped and formatted dataset
    """

    # Step 1: Load and reshape main database
    df1 = pd.read_csv(main_db_path)

    # Rename columns for readability
    df2 = df1.rename(columns={
        'AZIENDE SUDDIVISE PER CODICE ATECO': 'Name',
        'ATECO 2007\ncodice': 'ATECO',
        'ATECO 2007\ndescrizione': 'ATECOx'
    })

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

        # Add GRI, ESRS, SASB for each year
        for year in [2022, 2023, 2024]:
            year_data = company_data[company_data['Anno'] == year]
            if not year_data.empty:
                row_data[f'GRI_{year}'] = year_data['GRI'].iloc[0]
                row_data[f'ESRS_{year}'] = year_data['ESRS'].iloc[0]
                row_data[f'SASB_{year}'] = year_data['SASB'].iloc[0]
            else:
                row_data[f'GRI_{year}'] = 0
                row_data[f'ESRS_{year}'] = 0
                row_data[f'SASB_{year}'] = 0

        result_data.append(row_data)

    df = pd.DataFrame(result_data)

    # Handle missing values and data types
    df = df.replace('#N/A', np.nan)

    # Convert all indicator columns to integers
    indicator_cols = [col for col in df.columns if col.startswith(('GRI_', 'ESRS_', 'SASB_'))]
    for col in indicator_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Convert ATECO to appropriate type if needed
    df['ATECO'] = pd.to_numeric(df['ATECO'], errors='coerce')

    # Delete the last useless row
    df = df.drop(df.index[-1])

    # Step 2: Add 2-digit ATECO identifiers
    df_Extra = pd.read_csv(ateco_codes_path)
    df_Extra = df_Extra.rename(columns={
        'Codice Ateco': 'Codice',
        'Titolo Ateco 2007 aggiornamento 2022': 'Codice_desc'
    })
    df_Extra = df_Extra[['Codice', 'Codice_desc']]

    # Getting the values from the ISTAT database related to the general ateco codes
    ateco = list()
    atecoX = list()
    for index, row in df_Extra.iterrows():
        if len(row['Codice']) == 2:
            ateco.append(row['Codice'])
            atecoX.append(row['Codice_desc'])

    # Create mapping dictionary
    mapping_dict = dict(zip(ateco, atecoX))

    # Creating the new 2 columns which have the first 2 characters of ATECO
    df['Ateco'] = df['ATECO'].astype(str).str[:2]
    df['AtecoX'] = df['Ateco'].map(mapping_dict)

    # Putting the new 2 columns before the ATECO ones
    new_cols = df[['Ateco', 'AtecoX']]
    df = df.drop(df.columns[-2:], axis=1)  # Remove the temporary columns
    df = pd.concat([df.iloc[:, :1], new_cols, df.iloc[:, 1:]], axis=1)

    # Step 3: Process ATECO section codes and add 1-character ATECO sections
    # First, process the ATECO codes file to create section mapping
    df_Extra_processed = pd.read_csv(ateco_codes_path)
    df_Extra_processed = df_Extra_processed.rename(columns={
        'Codice Ateco': 'Codice',
        'Titolo Ateco 2007 aggiornamento 2022': 'Codice_desc'
    })
    df1_processed = df_Extra_processed[['Codice', 'Codice_desc']]

    # Process to clear descriptions for non-letter rows
    temp_file = "temp_ateco_processing.csv"
    df1_processed.to_csv(temp_file, index=False)

    rows = []
    with open(temp_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            codice = row['Codice'].strip()
            # Check if Codice is NOT a single letter (A-Z)
            if not re.match(r'^[A-Z]$', codice):
                row['Codice_desc'] = ''
            rows.append(row)

    processed_file = "processed_ateco_codes.csv"
    with open(processed_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Create mapping dictionary for ATECO codes to letters and descriptions
    step2_df = pd.read_csv(processed_file)
    ateco_mapping = {}
    current_letter = None
    current_desc = None

    for _, row in step2_df.iterrows():
        codice = str(row['Codice']).strip()
        codice_desc = str(row['Codice_desc']).strip()

        # If it's a letter row (A, B, C, etc.)
        if len(codice) == 1 and codice.isalpha():
            current_letter = codice
            current_desc = codice_desc
        # If it's a 2-digit code and we have a current letter
        elif len(codice) == 2 and codice.isdigit() and current_letter is not None:
            ateco_mapping[codice] = {
                'letter': current_letter,
                'description': current_desc
            }

    # Function to get ATECO letter and description based on 2-digit code
    def get_ateco_info(ateco_code):
        if pd.isna(ateco_code):
            return None, None

        # Convert to string and get first 2 digits
        code_str = str(ateco_code).split('.')[0]  # Handle decimal codes
        if len(code_str) >= 2:
            two_digit = code_str[:2]
            if two_digit in ateco_mapping:
                return ateco_mapping[two_digit]['letter'], ateco_mapping[two_digit]['description']

        return None, None

    # Apply the function to get ATECO letter and description
    df[['ateco_letter', 'ateco_description']] = df['Ateco'].apply(
        lambda x: pd.Series(get_ateco_info(x))
    )

    # Reorder columns to place the new columns after 'Name'
    cols = list(df.columns)
    name_idx = cols.index('Name')

    # Remove the new columns from their current position
    cols.remove('ateco_letter')
    cols.remove('ateco_description')

    # Insert them after 'Name'
    cols.insert(name_idx + 1, 'ateco_letter')
    cols.insert(name_idx + 2, 'ateco_description')

    df = df[cols]

    # Rename the columns as requested
    df = df.rename(columns={
        'ateco_letter': 'ateco',
        'ateco_description': 'atecoX'
    })

    # Final column ordering
    df = df[['Name', 'ateco', 'atecoX', 'Ateco', 'AtecoX', 'ATECO', 'ATECOx',
             'GRI_2022', 'ESRS_2022', 'SASB_2022',
             'GRI_2023', 'ESRS_2023', 'SASB_2023',
             'GRI_2024', 'ESRS_2024', 'SASB_2024']]

    # Clean up temporary files
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)
    if os.path.exists(processed_file):
        os.remove(processed_file)

    # Save the final dataset
    df.to_csv('Tidier_Dataset.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)

    print(f"Database reshaping complete!")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df

# ACTUAL CODE
df = reshape_complete_database("Database Ufficiale.csv", "ATECO_codes.csv")