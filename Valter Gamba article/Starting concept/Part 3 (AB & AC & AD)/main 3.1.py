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
    - ateco_codes_csv_path: Path to 'ATECO_codes.csv'
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

    df.to_csv('Starting_Dataset.csv.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)
    # Instead of saving, return the df
    return df

###################################

'''
df = comprehensive_database_reshaping(
    "Database Ufficiale.csv",
    "ATECO_codes.csv"
)
'''

#####################################

# Load the data
df = pd.read_csv('Tidier_Dataset.csv')

# Extract ATECO letter from the first column
df['ateco_letter'] = df['ateco'].str[0]

# Calculate number of ones for each company each year
for year in ['2022', '2023', '2024']:
    df[f'count_ones_{year}'] = df[[f'Field1_{year}', f'Field2_{year}', f'Field3_{year}']].sum(axis=1)

# Create detailed breakdown by ATECO letter and count of ones
detailed_breakdown = pd.DataFrame()

for year in ['2022', '2023', '2024']:
    year_data = df.groupby(['ateco_letter', f'count_ones_{year}']).size().unstack(fill_value=0)
    year_data = year_data.reindex(columns=[0, 1, 2, 3], fill_value=0)

    # Calculate total companies - only sum the count columns (0,1,2,3)
    year_data['total_companies'] = year_data[[0, 1, 2, 3]].sum(axis=1)
    year_data['year'] = year

    # Calculate percentages for each count
    for count in [0, 1, 2, 3]:
        if count in year_data.columns:
            year_data[f'pct_{count}'] = (year_data[count] / year_data['total_companies'] * 100).round(1)

    detailed_breakdown = pd.concat([detailed_breakdown, year_data])

# Reset index for easier handling
detailed_breakdown = detailed_breakdown.reset_index()

# Get sector order by total companies (for consistent ordering)
sector_order = df['ateco_letter'].value_counts().index

# Color scheme for 1, 2, 3 ones (excluding 0)
colors = ['#4ecdc4', '#45b7d1', '#96ceb4']  # teal, blue, green for 1, 2, 3 ones

# =============================================================================
# IMAGE 1: 2022 - PERCENTAGE AND ABSOLUTE VALUES
# =============================================================================
fig1, (ax1_pct, ax1_abs) = plt.subplots(1, 2, figsize=(20, 8))
fig1.suptitle('2022 - Distribution of Companies with at least one "1"', fontsize=16, fontweight='bold')

# Prepare 2022 data
year_2022_data = detailed_breakdown[detailed_breakdown['year'] == '2022'].set_index('ateco_letter')
year_2022_data = year_2022_data.reindex(sector_order)

# PERCENTAGE PLOT
bottom_pct = np.zeros(len(sector_order))
for count, color in zip([1, 2, 3], colors):
    pct_col = f'pct_{count}'
    if pct_col in year_2022_data.columns:
        values = year_2022_data[pct_col].values
        bars = ax1_pct.bar(year_2022_data.index, values, bottom=bottom_pct,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_pct += values

        # Add percentage labels for values > 5%
        for i, val in enumerate(values):
            if val > 5:
                ax1_pct.text(i, bottom_pct[i] - val / 2, f'{val:.0f}%',
                             ha='center', va='center', fontsize=8, fontweight='bold')

# Add total percentage line and labels
for i, total_pct in enumerate(bottom_pct):
    if total_pct > 0:
        ax1_pct.text(i, total_pct + 2, f'{total_pct:.0f}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

ax1_pct.set_title('Percentage Distribution', fontweight='bold', fontsize=14)
ax1_pct.set_ylabel('Percentage of Companies (%)')
ax1_pct.set_ylim(0, 110)
ax1_pct.tick_params(axis='x', rotation=45)
ax1_pct.grid(axis='y', alpha=0.3)
ax1_pct.legend(title='Number of 1s', loc='upper right')

# ABSOLUTE PLOT
bottom_abs = np.zeros(len(sector_order))
for count, color in zip([1, 2, 3], colors):
    if count in year_2022_data.columns:
        values = year_2022_data[count].values
        bars = ax1_abs.bar(year_2022_data.index, values, bottom=bottom_abs,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_abs += values

        # Add value labels for counts > 0
        for i, val in enumerate(values):
            if val > 0:
                ax1_abs.text(i, bottom_abs[i] - val / 2, f'{int(val)}',
                             ha='center', va='center', fontsize=8, fontweight='bold')

# Add total absolute line and labels
for i, total_abs in enumerate(bottom_abs):
    if total_abs > 0:
        ax1_abs.text(i, total_abs + 1, f'{int(total_abs)}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

ax1_abs.set_title('Absolute Counts', fontweight='bold', fontsize=14)
ax1_abs.set_ylabel('Number of Companies')
ax1_abs.tick_params(axis='x', rotation=45)
ax1_abs.grid(axis='y', alpha=0.3)
ax1_abs.legend(title='Number of 1s', loc='upper right')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig('Image1_2022.png', dpi=300, bbox_inches='tight')

# =============================================================================
# IMAGE 2: 2023 - PERCENTAGE AND ABSOLUTE VALUES
# =============================================================================
fig2, (ax2_pct, ax2_abs) = plt.subplots(1, 2, figsize=(20, 8))
fig2.suptitle('2023 - Distribution of Companies with at least one "1"', fontsize=16, fontweight='bold')

# Prepare 2023 data
year_2023_data = detailed_breakdown[detailed_breakdown['year'] == '2023'].set_index('ateco_letter')
year_2023_data = year_2023_data.reindex(sector_order)

# PERCENTAGE PLOT
bottom_pct = np.zeros(len(sector_order))
for count, color in zip([1, 2, 3], colors):
    pct_col = f'pct_{count}'
    if pct_col in year_2023_data.columns:
        values = year_2023_data[pct_col].values
        bars = ax2_pct.bar(year_2023_data.index, values, bottom=bottom_pct,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_pct += values

        # Add percentage labels for values > 5%
        for i, val in enumerate(values):
            if val > 5:
                ax2_pct.text(i, bottom_pct[i] - val / 2, f'{val:.0f}%',
                             ha='center', va='center', fontsize=8, fontweight='bold')

# Add total percentage line and labels
for i, total_pct in enumerate(bottom_pct):
    if total_pct > 0:
        ax2_pct.text(i, total_pct + 2, f'{total_pct:.0f}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

ax2_pct.set_title('Percentage Distribution', fontweight='bold', fontsize=14)
ax2_pct.set_ylabel('Percentage of Companies (%)')
ax2_pct.set_ylim(0, 110)
ax2_pct.tick_params(axis='x', rotation=45)
ax2_pct.grid(axis='y', alpha=0.3)
ax2_pct.legend(title='Number of 1s', loc='upper right')

# ABSOLUTE PLOT
bottom_abs = np.zeros(len(sector_order))
for count, color in zip([1, 2, 3], colors):
    if count in year_2023_data.columns:
        values = year_2023_data[count].values
        bars = ax2_abs.bar(year_2023_data.index, values, bottom=bottom_abs,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_abs += values

        # Add value labels for counts > 0
        for i, val in enumerate(values):
            if val > 0:
                ax2_abs.text(i, bottom_abs[i] - val / 2, f'{int(val)}',
                             ha='center', va='center', fontsize=8, fontweight='bold')

# Add total absolute line and labels
for i, total_abs in enumerate(bottom_abs):
    if total_abs > 0:
        ax2_abs.text(i, total_abs + 1, f'{int(total_abs)}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

ax2_abs.set_title('Absolute Counts', fontweight='bold', fontsize=14)
ax2_abs.set_ylabel('Number of Companies')
ax2_abs.tick_params(axis='x', rotation=45)
ax2_abs.grid(axis='y', alpha=0.3)
ax2_abs.legend(title='Number of 1s', loc='upper right')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig('Image2_2023.png', dpi=300, bbox_inches='tight')

# =============================================================================
# IMAGE 3: 2024 - PERCENTAGE AND ABSOLUTE VALUES
# =============================================================================
fig3, (ax3_pct, ax3_abs) = plt.subplots(1, 2, figsize=(20, 8))
fig3.suptitle('2024 - Distribution of Companies with at least one "1"', fontsize=16, fontweight='bold')

# Prepare 2024 data
year_2024_data = detailed_breakdown[detailed_breakdown['year'] == '2024'].set_index('ateco_letter')
year_2024_data = year_2024_data.reindex(sector_order)

# PERCENTAGE PLOT
bottom_pct = np.zeros(len(sector_order))
for count, color in zip([1, 2, 3], colors):
    pct_col = f'pct_{count}'
    if pct_col in year_2024_data.columns:
        values = year_2024_data[pct_col].values
        bars = ax3_pct.bar(year_2024_data.index, values, bottom=bottom_pct,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_pct += values

        # Add percentage labels for values > 5%
        for i, val in enumerate(values):
            if val > 5:
                ax3_pct.text(i, bottom_pct[i] - val / 2, f'{val:.0f}%',
                             ha='center', va='center', fontsize=8, fontweight='bold')

# Add total percentage line and labels
for i, total_pct in enumerate(bottom_pct):
    if total_pct > 0:
        ax3_pct.text(i, total_pct + 2, f'{total_pct:.0f}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

ax3_pct.set_title('Percentage Distribution', fontweight='bold', fontsize=14)
ax3_pct.set_ylabel('Percentage of Companies (%)')
ax3_pct.set_ylim(0, 110)
ax3_pct.tick_params(axis='x', rotation=45)
ax3_pct.grid(axis='y', alpha=0.3)
ax3_pct.legend(title='Number of 1s', loc='upper right')

# ABSOLUTE PLOT
bottom_abs = np.zeros(len(sector_order))
for count, color in zip([1, 2, 3], colors):
    if count in year_2024_data.columns:
        values = year_2024_data[count].values
        bars = ax3_abs.bar(year_2024_data.index, values, bottom=bottom_abs,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_abs += values

        # Add value labels for counts > 0
        for i, val in enumerate(values):
            if val > 0:
                ax3_abs.text(i, bottom_abs[i] - val / 2, f'{int(val)}',
                             ha='center', va='center', fontsize=8, fontweight='bold')

# Add total absolute line and labels
for i, total_abs in enumerate(bottom_abs):
    if total_abs > 0:
        ax3_abs.text(i, total_abs + 1, f'{int(total_abs)}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

ax3_abs.set_title('Absolute Counts', fontweight='bold', fontsize=14)
ax3_abs.set_ylabel('Number of Companies')
ax3_abs.tick_params(axis='x', rotation=45)
ax3_abs.grid(axis='y', alpha=0.3)
ax3_abs.legend(title='Number of 1s', loc='upper right')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig('Image3_2024.png', dpi=300, bbox_inches='tight')

# =============================================================================
# DETAILED ANALYSIS
# =============================================================================
print("=" * 80)
print("DETAILED ANALYSIS OF BINARY FIELD ENGAGEMENT ACROSS ATECO SECTORS (2022-2024)")
print("=" * 80)

# Calculate performance metrics
performance_analysis = pd.DataFrame()

for sector in sector_order:
    sector_data = {'Sector': sector, 'Total_Companies': len(df[df['ateco_letter'] == sector])}

    for year in ['2022', '2023', '2024']:
        year_data = detailed_breakdown[(detailed_breakdown['year'] == year) &
                                       (detailed_breakdown['ateco_letter'] == sector)]

        if not year_data.empty:
            row = year_data.iloc[0]
            # Use .loc to avoid integer indexing issues
            pct_0 = row.loc['pct_0'] if 'pct_0' in row.index else 0
            pct_3 = row.loc['pct_3'] if 'pct_3' in row.index else 0
            count_1 = row.loc[1] if 1 in row.index else 0
            count_2 = row.loc[2] if 2 in row.index else 0
            count_3 = row.loc[3] if 3 in row.index else 0

            sector_data[f'Pct_1plus_{year}'] = 100 - pct_0  # Percentage with at least one 1
            sector_data[f'Abs_1plus_{year}'] = count_1 + count_2 + count_3  # Absolute with at least one 1
            sector_data[f'Pct_3ones_{year}'] = pct_3  # Percentage with 3 ones
            sector_data[f'Abs_3ones_{year}'] = count_3  # Absolute with 3 ones

    performance_analysis = pd.concat([performance_analysis, pd.DataFrame([sector_data])], ignore_index=True)

# Calculate trends
performance_analysis['Trend_1plus_Pct'] = performance_analysis['Pct_1plus_2024'] - performance_analysis[
    'Pct_1plus_2022']
performance_analysis['Trend_3ones_Pct'] = performance_analysis['Pct_3ones_2024'] - performance_analysis[
    'Pct_3ones_2022']
performance_analysis['Avg_Pct_1plus'] = performance_analysis[
    ['Pct_1plus_2022', 'Pct_1plus_2023', 'Pct_1plus_2024']].mean(axis=1)

print("\n1. SECTOR PERFORMANCE CLASSIFICATION")
print("-" * 50)

# Classify sectors by performance
high_performers = performance_analysis[performance_analysis['Avg_Pct_1plus'] > 60]
medium_performers = performance_analysis[(performance_analysis['Avg_Pct_1plus'] >= 30) &
                                         (performance_analysis['Avg_Pct_1plus'] <= 60)]
low_performers = performance_analysis[performance_analysis['Avg_Pct_1plus'] < 30]

print(f"High Performers (>60% average engagement): {len(high_performers)} sectors")
for _, row in high_performers.sort_values('Avg_Pct_1plus', ascending=False).iterrows():
    print(f"  {row['Sector']}: {row['Avg_Pct_1plus']:.1f}% avg, {row['Total_Companies']} companies")

print(f"\nMedium Performers (30-60% average engagement): {len(medium_performers)} sectors")
for _, row in medium_performers.sort_values('Avg_Pct_1plus', ascending=False).iterrows():
    print(f"  {row['Sector']}: {row['Avg_Pct_1plus']:.1f}% avg, {row['Total_Companies']} companies")

print(f"\nLow Performers (<30% average engagement): {len(low_performers)} sectors")
for _, row in low_performers.sort_values('Avg_Pct_1plus', ascending=False).iterrows():
    print(f"  {row['Sector']}: {row['Avg_Pct_1plus']:.1f}% avg, {row['Total_Companies']} companies")

print("\n2. TREND ANALYSIS (2022-2024)")
print("-" * 50)

# Biggest improvers
biggest_improvers = performance_analysis.nlargest(5, 'Trend_1plus_Pct')
print("Biggest Improvements in Engagement:")
for _, row in biggest_improvers.iterrows():
    print(f"  {row['Sector']}: +{row['Trend_1plus_Pct']:.1f}% "
          f"({row['Pct_1plus_2022']:.1f}% → {row['Pct_1plus_2024']:.1f}%)")

# Biggest decliners
biggest_decliners = performance_analysis.nsmallest(5, 'Trend_1plus_Pct')
print("\nBiggest Declines in Engagement:")
for _, row in biggest_decliners.iterrows():
    if row['Trend_1plus_Pct'] < 0:
        print(f"  {row['Sector']}: {row['Trend_1plus_Pct']:.1f}% "
              f"({row['Pct_1plus_2022']:.1f}% → {row['Pct_1plus_2024']:.1f}%)")

print("\n3. EXCELLENCE ANALYSIS (Companies with 3 '1s')")
print("-" * 50)

# Sectors with highest percentage of companies achieving 3 ones in 2024
excellence_leaders = performance_analysis.nlargest(5, 'Pct_3ones_2024')
print("Sectors with Highest Excellence (3 '1s') in 2024:")
for _, row in excellence_leaders.iterrows():
    if row['Pct_3ones_2024'] > 0:
        print(f"  {row['Sector']}: {row['Pct_3ones_2024']:.1f}% ({row['Abs_3ones_2024']} companies)")

print("\n4. SECTOR SIZE VS ENGAGEMENT ANALYSIS")
print("-" * 50)

# Large sectors with good performance
large_engaged = performance_analysis[
    (performance_analysis['Total_Companies'] > 20) &
    (performance_analysis['Avg_Pct_1plus'] > 40)
    ].sort_values('Total_Companies', ascending=False)

print("Large Sectors (>20 companies) with Good Engagement (>40%):")
for _, row in large_engaged.iterrows():
    print(f"  {row['Sector']}: {row['Total_Companies']} companies, {row['Avg_Pct_1plus']:.1f}% avg engagement")

# Small sectors with exceptional performance
small_excellent = performance_analysis[
    (performance_analysis['Total_Companies'] <= 5) &
    (performance_analysis['Avg_Pct_1plus'] > 70)
    ].sort_values('Avg_Pct_1plus', ascending=False)

print("\nSmall Sectors (≤5 companies) with Exceptional Performance (>70%):")
for _, row in small_excellent.iterrows():
    print(f"  {row['Sector']}: {row['Total_Companies']} companies, {row['Avg_Pct_1plus']:.1f}% avg engagement")

print("\n5. KEY INSIGHTS AND RECOMMENDATIONS")
print("-" * 50)

# Final summary table
print("\n6. PERFORMANCE SUMMARY TABLE (2024)")
print("-" * 50)
summary_2024 = performance_analysis[
    ['Sector', 'Total_Companies', 'Pct_1plus_2024', 'Pct_3ones_2024', 'Trend_1plus_Pct']].copy()
summary_2024 = summary_2024.sort_values('Pct_1plus_2024', ascending=False)
summary_2024.columns = ['Sector', 'Total Cos', '2024: % ≥1', '2024: % 3', 'Trend 22-24']
print(summary_2024.round(1).to_string(index=False))