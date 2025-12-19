import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt


def comprehensive_database_reshaping(original_csv_path, ateco_codes_csv_path, output_csv_path='Starting_Dataset.csv',
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
    - output_csv_path: Path to save the final CSV (default: 'Starting_Dataset.csv')
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
        "APPLICAZIONE STANDARD AA1000": "AA1000",
        "SUDDIVISIONE STAKEHOLDER INTERNI/ESTERNI" : "Field1",
        "DESCRIZIONE PROCESSO DI ATTRIBUZIONE DI RILEVANZA DEGLI STAKEHOLDER" : "Field2",
        "DESCRIZIONE PROCESSO DI COINVOLGIMENTO" : "Field3"
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
                row_data[f'AA1000_{year}'] = year_data['AA1000'].iloc[0]
                row_data[f'Field1_{year}'] = year_data['Field1'].iloc[0]
                row_data[f'Field2_{year}'] = year_data['Field2'].iloc[0]
                row_data[f'Field3_{year}'] = year_data['Field3'].iloc[0]
            else:
                row_data[f'AA1000_{year}'] = 0
                row_data[f'Field1_{year}'] = 0
                row_data[f'Field2_{year}'] = 0
                row_data[f'Field3_{year}'] = 0
        result_data.append(row_data)
    df = pd.DataFrame(result_data)
    df = df.replace('#N/A', np.nan)
    #CAN CHANGE
    indicator_cols = [col for col in df.columns if col.startswith(('AA1000_', 'Field1_', 'Field2_', 'Field3_'))]
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
#THIS CAN CHANGE
    df = df.drop(df.columns[15:], axis=1)
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

    # Reorder columns
    df = df[['Name', 'ateco', 'atecoX', 'Ateco', 'AtecoX', 'ATECO', 'ATECOx',
             'AA1000_2022', 'AA1000_2023', 'AA1000_2024',
             'Field1_2022', 'Field1_2023', 'Field1_2024',
             'Field2_2022', 'Field2_2023', 'Field2_2024',
             'Field3_2022', 'Field3_2023', 'Field3_2024',]]

    #Remove AA1000 columns
    all_columns = df.columns.tolist()
    columns_to_keep = [col for col in all_columns if col not in ['AA1000_2022', 'AA1000_2023', 'AA1000_2024']]
    df = df[columns_to_keep]

    df.to_csv('Starting_Dataset.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)
    # Instead of saving, return the df
    return df


df = comprehensive_database_reshaping(
    "Database Ufficiale.csv",
    "ATECO_codes.csv"
)

# Load the data
df = pd.read_csv('Tidier_Dataset.csv')

# Extract ATECO letter from the first column (it's the first character)
df['ateco_letter'] = df['ateco'].str[0]

# Create binary indicators for each year - 1 if any field has 1, 0 otherwise
df['has_ones_2022'] = ((df['Field1_2022'] == 1) | (df['Field2_2022'] == 1) | (df['Field3_2022'] == 1)).astype(int)
df['has_ones_2023'] = ((df['Field1_2023'] == 1) | (df['Field2_2023'] == 1) | (df['Field3_2023'] == 1)).astype(int)
df['has_ones_2024'] = ((df['Field1_2024'] == 1) | (df['Field2_2024'] == 1) | (df['Field3_2024'] == 1)).astype(int)

# Calculate totals per ATECO letter for each year
yearly_totals = df.groupby('ateco_letter').agg({
    'has_ones_2022': 'sum',
    'has_ones_2023': 'sum',
    'has_ones_2024': 'sum',
    'Name': 'count'  # Total companies per sector
}).rename(columns={'Name': 'total_companies'})

# Calculate percentages
for year in ['2022', '2023', '2024']:
    yearly_totals[f'percentage_{year}'] = (yearly_totals[f'has_ones_{year}'] / yearly_totals['total_companies'] * 100).round(1)

# Sort by total companies for better visualization
yearly_totals = yearly_totals.sort_values('total_companies', ascending=False)

print("COMPANIES WITH AT LEAST ONE '1' BY ATECO SECTOR")
print("="*50)
print(yearly_totals[['total_companies', 'has_ones_2022', 'has_ones_2023', 'has_ones_2024',
                   'percentage_2022', 'percentage_2023', 'percentage_2024']])

# Create histograms
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Distribution of Companies with at least one "1" by ATECO Sector (2022-2024)', fontsize=16, fontweight='bold')

# Year 2022
axes[0,0].bar(yearly_totals.index, yearly_totals['has_ones_2022'], color='skyblue', alpha=0.7)
axes[0,0].set_title('2022 - Companies with ≥1 "1"', fontweight='bold')
axes[0,0].set_ylabel('Number of Companies')
axes[0,0].tick_params(axis='x', rotation=45)

# Year 2023
axes[0,1].bar(yearly_totals.index, yearly_totals['has_ones_2023'], color='lightgreen', alpha=0.7)
axes[0,1].set_title('2023 - Companies with ≥1 "1"', fontweight='bold')
axes[0,1].set_ylabel('Number of Companies')
axes[0,1].tick_params(axis='x', rotation=45)

# Year 2024
axes[1,0].bar(yearly_totals.index, yearly_totals['has_ones_2024'], color='salmon', alpha=0.7)
axes[1,0].set_title('2024 - Companies with ≥1 "1"', fontweight='bold')
axes[1,0].set_ylabel('Number of Companies')
axes[1,0].tick_params(axis='x', rotation=45)

# Total across all years
total_ones = yearly_totals['has_ones_2022'] + yearly_totals['has_ones_2023'] + yearly_totals['has_ones_2024']
axes[1,1].bar(yearly_totals.index, total_ones, color='gold', alpha=0.7)
axes[1,1].set_title('Total "1"s across 2022-2024', fontweight='bold')
axes[1,1].set_ylabel('Cumulative Count')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('INSERIRE TITOLO ADEGUATO (2022-2024).png')

# Additional analysis: High performers vs Low performers
print("\n" + "="*60)
print("SECTOR PERFORMANCE ANALYSIS")
print("="*60)

# Calculate average percentage across years
yearly_totals['avg_percentage'] = yearly_totals[['percentage_2022', 'percentage_2023', 'percentage_2024']].mean(axis=1)

# Define performance tiers
high_performers = yearly_totals[yearly_totals['avg_percentage'] > 60]
medium_performers = yearly_totals[(yearly_totals['avg_percentage'] >= 30) & (yearly_totals['avg_percentage'] <= 60)]
low_performers = yearly_totals[yearly_totals['avg_percentage'] < 30]

print(f"\nHIGH PERFORMERS (Avg > 60% companies with ones):")
print(high_performers[['total_companies', 'avg_percentage']].sort_values('avg_percentage', ascending=False))

print(f"\nMEDIUM PERFORMERS (30-60% companies with ones):")
print(medium_performers[['total_companies', 'avg_percentage']].sort_values('avg_percentage', ascending=False))

print(f"\nLOW PERFORMERS (<30% companies with ones):")
print(low_performers[['total_companies', 'avg_percentage']].sort_values('avg_percentage', ascending=False))

# Trend analysis
print(f"\n" + "="*60)
print("TREND ANALYSIS 2022-2024")
print("="*60)

trend_analysis = []
for letter in yearly_totals.index:
    pct_2022 = yearly_totals.loc[letter, 'percentage_2022']
    pct_2024 = yearly_totals.loc[letter, 'percentage_2024']
    trend = "↑ Improving" if pct_2024 > pct_2022 + 5 else "↓ Declining" if pct_2024 < pct_2022 - 5 else "→ Stable"
    trend_analysis.append({
        'Sector': letter,
        '2022': f"{pct_2022}%",
        '2024': f"{pct_2024}%",
        'Trend': trend,
        'Change': f"{(pct_2024 - pct_2022):+.1f}%"
    })

trend_df = pd.DataFrame(trend_analysis)
print(trend_df)