import numpy as np
import pandas as pd
import re


def comprehensive_database_reshaping(original_csv_path, ateco_codes_csv_path, output_csv_path='Tidier_Dataset.csv',
                                     output_excel_path='Tidier_Dataset.xlsx'):
    """
    Comprehensive function to perform all database reshaping steps in a single function:
    1. Read and reshape the original database.
    2. Add new ATECO identifiers.
    3. Process ATECO codes to create intermediate files.
    4. Map ATECO letters and descriptions.
    5. Reorder columns.
    6. Save the final tidied dataset.

    Parameters:
    - original_csv_path: Path to 'Database Ufficiale.csv'
    - ateco_codes_csv_path: Path to 'ATECO_codes.csv'
    - output_csv_path: Path to save the final CSV (default: 'Tidier_Dataset.csv')
    - output_excel_path: Path to save the final Excel (default: 'Tidier_Dataset.xlsx')

    Returns:
    - Final tidied DataFrame
    """

    # Read original CSV
    df1 = pd.read_csv(original_csv_path)

    # CAN CHANGE
    # Reshape the database
    df2 = df1.rename(columns={
        'AZIENDE SUDDIVISE PER CODICE ATECO': 'Name',
        'ATECO 2007\r\ncodice': 'ATECO',
        'ATECO 2007\r\ndescrizione': 'ATECOx',
        'INTERVISTE' : 'Field1',
        'SURVEY ONLINE' : 'Field2',
        'CONFERENCE CALL' : 'Field3',
        'VISITE IN LOCO': 'Field4',
        'SEMINARI ' : 'Field5',
        'CONFERENZE': 'Field6',
        'RAPPORTI CON AGRICOLTORI': 'Field7',
        'ROADSHOW': 'Field8',
        'QUESTIONARI': 'Field9',
        'FOCUS GROUP': 'Field10',
        'WORKSHOP': 'Field11',
        'GROUP MEETING': 'Field12',
        'INCONTRI PERIODICI': 'Field13',
        'INTRANET AZIENDALE ': 'Field14',
        'NEWSLETTERS': 'Field15',
        'COMUNICATI STAMPA': 'Field16'

    })

    # extra
    #print(df2.columns)

    companies = df2['Name'].unique()
    result_data = []
    for company in companies:
        company_data = df2[df2['Name'] == company]
        base_info = company_data.iloc[0][['Name', 'ATECO', 'ATECOx']]
        row_data = {
            'Name': base_info['Name'],
            'ATECO': base_info['ATECO'],
            'ATECOx': base_info['ATECOx']
        }
        # CAN CHANGE
        for year in [2022, 2023, 2024]:
            year_data = company_data[company_data['Anno'] == year]
            if not year_data.empty:
                row_data[f'Field1_{year}'] = year_data['Field1'].iloc[0]
                row_data[f'Field2_{year}'] = year_data['Field2'].iloc[0]
                row_data[f'Field3_{year}'] = year_data['Field3'].iloc[0]
                row_data[f'Field4_{year}'] = year_data['Field4'].iloc[0]
                row_data[f'Field5_{year}'] = year_data['Field5'].iloc[0]
                row_data[f'Field6_{year}'] = year_data['Field6'].iloc[0]
                row_data[f'Field7_{year}'] = year_data['Field7'].iloc[0]
                row_data[f'Field8_{year}'] = year_data['Field8'].iloc[0]
                row_data[f'Field9_{year}'] = year_data['Field9'].iloc[0]
                row_data[f'Field10_{year}'] = year_data['Field10'].iloc[0]
                row_data[f'Field11_{year}'] = year_data['Field11'].iloc[0]
                row_data[f'Field12_{year}'] = year_data['Field12'].iloc[0]
                row_data[f'Field13_{year}'] = year_data['Field13'].iloc[0]
                row_data[f'Field14_{year}'] = year_data['Field14'].iloc[0]
                row_data[f'Field15_{year}'] = year_data['Field15'].iloc[0]
                row_data[f'Field16_{year}'] = year_data['Field16'].iloc[0]
            else:
                row_data[f'Field1_{year}'] = 0
                row_data[f'Field2_{year}'] = 0
                row_data[f'Field3_{year}'] = 0
                row_data[f'Field4_{year}'] = 0
                row_data[f'Field5_{year}'] = 0
                row_data[f'Field6_{year}'] = 0
                row_data[f'Field7_{year}'] = 0
                row_data[f'Field8_{year}'] = 0
                row_data[f'Field9_{year}'] = 0
                row_data[f'Field10_{year}'] = 0
                row_data[f'Field11_{year}'] = 0
                row_data[f'Field12_{year}'] = 0
                row_data[f'Field13_{year}'] = 0
                row_data[f'Field14_{year}'] = 0
                row_data[f'Field15_{year}'] = 0
                row_data[f'Field16_{year}'] = 0
        result_data.append(row_data)
    df = pd.DataFrame(result_data)
    df = df.replace('#N/A', np.nan)
    #CAN CHANGE
    indicator_cols = [col for col in df.columns if col.startswith((
        'Field1_', 'Field2_', 'Field3_', 'Field4_', 'Field5_', 'Field6_', 'Field7_', 'Field8_',
        'Field9_', 'Field10_', 'Field11_', 'Field12_', 'Field13_', 'Field14_', 'Field15_', 'Field16_'))]
    for col in indicator_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['ATECO'] = pd.to_numeric(df['ATECO'], errors='coerce').fillna(0).astype(int)
    df = df.drop(df.index[-1])  # Delete the last useless row

    # Add new ATECO identifiers
    df_Extra = pd.read_csv(ateco_codes_csv_path)
    df_Extra = df_Extra.rename(columns={
        'Codice Ateco': 'Codice',
        'Titolo Ateco 2007 aggiornamento 2022': 'Codice_desc'}
    )
    df_Extra = df_Extra[['Codice', 'Codice_desc']]
    ateco = []
    atecoX = []
    for index, row in df_Extra.iterrows():
        if len(row['Codice']) == 2:
            ateco.append(row['Codice'])
            atecoX.append(row['Codice_desc'])
    mapping_dict = dict(zip(ateco, atecoX))
    df['Ateco'] = df['ATECO'].astype(str).str[:2]
    df['AtecoX'] = df['Ateco'].map(mapping_dict)
    new_cols = df[['Ateco', 'AtecoX']]
#THIS CAN CHANGE: 3+#(added columns)
    df = df.drop(df.columns[51:], axis=1)
    df = pd.concat([df.iloc[:, :1], new_cols, df.iloc[:, 1:]], axis=1)

    # Process ATECO codes (step1) in memory
    df_Extra = pd.read_csv(ateco_codes_csv_path)
    df_Extra = df_Extra.rename(columns={
        'Codice Ateco': 'Codice',
        'Titolo Ateco 2007 aggiornamento 2022': 'Codice_desc'}
    )
    df1_step = df_Extra[['Codice', 'Codice_desc']]

    # Modify in memory instead of creating step2.csv
    def modify_row(row):
        codice = str(row['Codice']).strip()
        if not re.match(r'^[A-Z]$', codice):
            row['Codice_desc'] = ''
        return row

    modified_df = df1_step.apply(modify_row, axis=1)

    # Map ATECO letters and descriptions (step2)
    ateco_mapping = {}
    current_letter = None
    current_desc = None
    for _, row in modified_df.iterrows():
        codice = str(row['Codice']).strip()
        codice_desc = str(row['Codice_desc']).strip()
        if len(codice) == 1 and codice.isalpha():
            current_letter = codice
            current_desc = codice_desc
        elif len(codice) == 2 and codice.isdigit() and current_letter is not None:
            ateco_mapping[codice] = {
                'letter': current_letter,
                'description': current_desc
            }

    def get_ateco_info(ateco_code):
        if pd.isna(ateco_code):
            return None, None
        code_str = str(ateco_code).split('.')[0]
        if len(code_str) >= 2:
            two_digit = code_str[:2]
            if two_digit in ateco_mapping:
                return ateco_mapping[two_digit]['letter'], ateco_mapping[two_digit]['description']
        return None, None

    df[['ateco_letter', 'ateco_description']] = df['Ateco'].apply(
        lambda x: pd.Series(get_ateco_info(x))
    )
    cols = list(df.columns)
    name_idx = cols.index('Name')
    cols.remove('ateco_letter')
    cols.remove('ateco_description')
    cols.insert(name_idx + 1, 'ateco_letter')
    cols.insert(name_idx + 2, 'ateco_description')
    df = df[cols]
    df = df.rename(columns={
        'ateco_letter': 'ateco',
        'ateco_description': 'atecoX'
    })

    #CAN CHANGE
    # Reorder columns
    df = df[['Name', 'ateco', 'atecoX', 'Ateco', 'AtecoX', 'ATECO', 'ATECOx',
             'Field1_2022', 'Field1_2023', 'Field1_2024',
             'Field2_2022', 'Field2_2023', 'Field2_2024',
             'Field3_2022', 'Field3_2023', 'Field3_2024',
             'Field4_2022', 'Field4_2023', 'Field4_2024',
             'Field5_2022', 'Field5_2023', 'Field5_2024',
             'Field6_2022', 'Field6_2023', 'Field6_2024',
             'Field7_2022', 'Field7_2023', 'Field7_2024',
             'Field8_2022', 'Field8_2023', 'Field8_2024',
             'Field9_2022', 'Field9_2023', 'Field9_2024',
             'Field10_2022', 'Field10_2023', 'Field10_2024',
             'Field11_2022', 'Field11_2023', 'Field11_2024',
             'Field12_2022', 'Field12_2023', 'Field12_2024',
             'Field13_2022', 'Field13_2023', 'Field13_2024',
             'Field14_2022', 'Field14_2023', 'Field14_2024',
             'Field15_2022', 'Field15_2023', 'Field15_2024',
             'Field16_2022', 'Field16_2023', 'Field16_2024'
             ]]

    df.to_csv('Tidier_Dataset.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)
    # Instead of saving, return the df
    return df


df = comprehensive_database_reshaping(
    "Database Ufficiale.csv",
    "ATECO_codes.csv"
)

# Print the final dataset
print(df.columns)