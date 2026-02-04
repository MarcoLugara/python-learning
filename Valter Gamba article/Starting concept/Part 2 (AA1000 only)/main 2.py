import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt


def comprehensive_database_reshaping(original_csv_path, ateco_codes_csv_path, output_csv_path='Starting_Dataset.csv.csv',
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
    - ateco_codes_csv_path: Path to 'ATECLO_codes.csv'
    - output_csv_path: Path to save the final CSV (default: 'Starting_Dataset.csv.csv')
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
        "APPLICAZIONE STANDARD AA1000": "AA1000"
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
            else:
                row_data[f'AA1000_{year}'] = 0
        result_data.append(row_data)
    df = pd.DataFrame(result_data)
    df = df.replace('#N/A', np.nan)
    #CAN CHANGE
    indicator_cols = [col for col in df.columns if col.startswith(('AA1000_'))]
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
    df = df.drop(df.columns[6:], axis=1)
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
             'AA1000_2022', 'AA1000_2023', 'AA1000_2024']]

    df.to_csv('Starting_Dataset.csv.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)
    # Instead of saving, return the df
    return df

#CODE LEFTOVERS
df = comprehensive_database_reshaping(
    "Database Ufficiale.csv",
    "ATECO_codes.csv"
)

#ACTUAL CODE
path = "Tidier_Dataset.csv"
df = pd.read_csv(path)

# Filter only companies with at least one year of AA1000 certification
df_with_aa1000 = df[(df['AA1000_2022'] == 1) | (df['AA1000_2023'] == 1) | (df['AA1000_2024'] == 1)].copy()


# Create year combination categories
def get_year_combination(row):
    years = []
    if row['AA1000_2022'] == 1:
        years.append('2022')
    if row['AA1000_2023'] == 1:
        years.append('2023')
    if row['AA1000_2024'] == 1:
        years.append('2024')

    if len(years) == 3:
        return 'All three years'
    elif len(years) == 2:
        return f"{years[0]} & {years[1]}"
    else:
        return years[0]


df_with_aa1000['year_combination'] = df_with_aa1000.apply(get_year_combination, axis=1)

# Group by sector and year combination
sector_year_analysis = pd.crosstab(df_with_aa1000['ateco'], df_with_aa1000['year_combination'])

# Define a color scheme for different year combinations
year_combinations = ['2022', '2023', '2024', '2022 & 2023', '2022 & 2024', '2023 & 2024', 'All three years']
colors = {
    '2022': '#ff6b6b',
    '2023': '#4ecdc4',
    '2024': '#45b7d1',
    '2022 & 2023': '#96ceb4',
    '2022 & 2024': '#feca57',
    '2023 & 2024': '#ff9ff3',
    'All three years': '#54a0ff'
}

# Create the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Create stacked bar chart
bottom_values = np.zeros(len(sector_year_analysis))
bar_width = 0.8

for year_combo in year_combinations:
    if year_combo in sector_year_analysis.columns:
        values = sector_year_analysis[year_combo]
        ax.bar(sector_year_analysis.index, values, bottom=bottom_values,
               label=year_combo, color=colors[year_combo], alpha=0.8, width=bar_width)
        bottom_values += values

# Customize the plot
ax.set_xlabel('Sector (ATECO Code)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Companies', fontsize=12, fontweight='bold')
ax.set_title(
    'Companies with AA1000 Certification by Sector and Year Combination\n(Companies with at least 1 year of certification)',
    fontsize=14, fontweight='bold', pad=20)

# Add legend
ax.legend(title='Year Combinations', title_fontsize=11, fontsize=9,
          bbox_to_anchor=(1.05, 1), loc='upper left')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value labels on bars
for i, sector in enumerate(sector_year_analysis.index):
    total_height = 0
    for year_combo in year_combinations:
        if year_combo in sector_year_analysis.columns:
            value = sector_year_analysis.loc[sector, year_combo]
            if value > 0:
                total_height += value
                ax.text(i, total_height - value / 2, str(int(value)),
                        ha='center', va='center', fontweight='bold', fontsize=8)

# Adjust layout
plt.tight_layout()
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Show the plot
plt.savefig('INSERIRE TITOLO ADEGUATO (2022-2024).png')

# Print detailed summary statistics
print("=" * 70)
print("DETAILED SUMMARY STATISTICS BY YEAR COMBINATION")
print("=" * 70)
print(f"Total companies in dataset: {len(df)}")
print(f"Companies with at least 1 year of AA1000: {len(df_with_aa1000)}")
print(f"\nBreakdown by year combination:")
for year_combo in year_combinations:
    count = len(df_with_aa1000[df_with_aa1000['year_combination'] == year_combo])
    if count > 0:
        print(f"  - {year_combo}: {count} companies")

print(f"\nSector analysis:")
for sector in sector_year_analysis.index:
    sector_data = sector_year_analysis.loc[sector]
    total = sector_data.sum()
    if total > 0:
        breakdown = []
        for year_combo in year_combinations:
            if year_combo in sector_year_analysis.columns:
                count = sector_data[year_combo]
                if count > 0:
                    breakdown.append(f"{year_combo}({int(count)})")
        print(f"  - Sector {sector}: {int(total)} companies")
        print(f"      Details: {', '.join(breakdown)}")

# Additional analysis: Year-over-year trends
print(f"\nYEAR-OVER-YEAR TRENDS:")
print(f"  - Companies with AA1000 in 2022: {(df['AA1000_2022'] == 1).sum()}")
print(f"  - Companies with AA1000 in 2023: {(df['AA1000_2023'] == 1).sum()}")
print(f"  - Companies with AA1000 in 2024: {(df['AA1000_2024'] == 1).sum()}")

# Calculate retention/dropoff
aa1000_2022 = df[df['AA1000_2022'] == 1]
aa1000_2023 = df[df['AA1000_2023'] == 1]
aa1000_2024 = df[df['AA1000_2024'] == 1]

continued_2022_to_2023 = len(aa1000_2022[aa1000_2022['AA1000_2023'] == 1])
continued_2023_to_2024 = len(aa1000_2023[aa1000_2023['AA1000_2024'] == 1])

print(f"  - Companies continuing from 2022 to 2023: {continued_2022_to_2023}")
print(f"  - Companies continuing from 2023 to 2024: {continued_2023_to_2024}")