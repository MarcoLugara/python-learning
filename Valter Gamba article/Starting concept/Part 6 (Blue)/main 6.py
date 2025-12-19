import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import datetime
import os


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

    #CAN CHANGE
    # Reshape the database
    df2 = df1.rename(columns={
        'AZIENDE SUDDIVISE PER CODICE ATECO': 'Name',
        'ATECO 2007\r\ncodice': 'ATECO',
        'ATECO 2007\r\ndescrizione': 'ATECOx',
        'INTRANET AZIENDALE ' : 'Field1',
        'NEWSLETTERS' : 'Field2',
        'COMUNICATI STAMPA' : 'Field3'
    })

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
        #CAN CHANGE
        for year in [2022, 2023, 2024]:
            year_data = company_data[company_data['Anno'] == year]
            if not year_data.empty:
                row_data[f'Field1_{year}'] = year_data['Field1'].iloc[0]
                row_data[f'Field2_{year}'] = year_data['Field2'].iloc[0]
                row_data[f'Field3_{year}'] = year_data['Field3'].iloc[0]
            else:
                row_data[f'Field1_{year}'] = 0
                row_data[f'Field2_{year}'] = 0
                row_data[f'Field3_{year}'] = 0
        result_data.append(row_data)
    df = pd.DataFrame(result_data)
    df = df.replace('#N/A', np.nan)
    #CAN CHANGE
    indicator_cols = [col for col in df.columns if col.startswith(('Field1_', 'Field2_', 'Field3_'))]
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
#THIS CAN CHANGE: 3+#(added columns*3)
    df = df.drop(df.columns[12:], axis=1)
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
             'Field3_2022', 'Field3_2023', 'Field3_2024'
             ]]

    df.to_csv('Starting_Dataset.csv.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)
    # Instead of saving, return the df
    return df


'''
df = comprehensive_database_reshaping(
    "Database Ufficiale.csv",
    "ATECO_codes.csv"
)
'''

# =============================================================================
# MAIN ANALYSIS CODE
# =============================================================================

# Load the data from your provided CSV
df = pd.read_csv('Tidier_Dataset.csv')

# Extract ATECO letter from the first column
df['ateco_letter'] = df['ateco'].str[0]

# Calculate number of ones for each company each year (3 fields)
for year in ['2022', '2023', '2024']:
    df[f'count_ones_{year}'] = df[[f'Field1_{year}', f'Field2_{year}', f'Field3_{year}']].sum(axis=1)

# Create detailed breakdown by ATECO letter and count of ones
detailed_breakdown = pd.DataFrame()

for year in ['2022', '2023', '2024']:
    year_data = df.groupby(['ateco_letter', f'count_ones_{year}']).size().unstack(fill_value=0)
    year_data = year_data.reindex(columns=list(range(0, 4)), fill_value=0)  # 0 to 3 ones

    # Calculate total companies - only sum the count columns (0-3)
    year_data['total_companies'] = year_data[list(range(0, 4))].sum(axis=1)
    year_data['year'] = year

    # Calculate percentages for each count
    for count in range(0, 4):  # 0 to 3
        if count in year_data.columns:
            year_data[f'pct_{count}'] = (year_data[count] / year_data['total_companies'] * 100).round(1)

    detailed_breakdown = pd.concat([detailed_breakdown, year_data])

# Reset index for easier handling
detailed_breakdown = detailed_breakdown.reset_index()

# Get sector order by total companies (for consistent ordering)
sector_order = df['ateco_letter'].value_counts().index

# Color scheme for 1-3 ones (excluding 0)
bar_colors = ['#4ecdc4', '#45b7d1', '#96ceb4'][:3]  # 3 colors

# =============================================================================
# IMAGE 1: 2022 - PERCENTAGE AND ABSOLUTE VALUES
# =============================================================================
fig1, (ax1_pct, ax1_abs) = plt.subplots(1, 2, figsize=(24, 8))
fig1.suptitle('2022 - Distribution of Companies with at least one "1" (3 Fields)', fontsize=16, fontweight='bold')

# Prepare 2022 data
year_2022_data = detailed_breakdown[detailed_breakdown['year'] == '2022'].set_index('ateco_letter')
year_2022_data = year_2022_data.reindex(sector_order)

# PERCENTAGE PLOT
bottom_pct = np.zeros(len(sector_order))
for count, color in zip(range(1, 4), bar_colors):  # 1 to 3
    pct_col = f'pct_{count}'
    if pct_col in year_2022_data.columns:
        values = year_2022_data[pct_col].values
        bars = ax1_pct.bar(year_2022_data.index, values, bottom=bottom_pct,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_pct += values

        # Add percentage labels for values > 3%
        for i, val in enumerate(values):
            if val > 3:
                ax1_pct.text(i, bottom_pct[i] - val / 2, f'{val:.0f}%',
                             ha='center', va='center', fontsize=7, fontweight='bold')

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
ax1_pct.legend(title='Number of 1s', loc='upper right', fontsize='small')

# ABSOLUTE PLOT
bottom_abs = np.zeros(len(sector_order))
for count, color in zip(range(1, 4), bar_colors):  # 1 to 3
    if count in year_2022_data.columns:
        values = year_2022_data[count].values
        bars = ax1_abs.bar(year_2022_data.index, values, bottom=bottom_abs,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_abs += values

        # Add value labels for counts > 0
        for i, val in enumerate(values):
            if val > 0:
                ax1_abs.text(i, bottom_abs[i] - val / 2, f'{int(val)}',
                             ha='center', va='center', fontsize=7, fontweight='bold')

# Add total absolute line and labels
for i, total_abs in enumerate(bottom_abs):
    if total_abs > 0:
        ax1_abs.text(i, total_abs + 1, f'{int(total_abs)}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

ax1_abs.set_title('Absolute Counts', fontweight='bold', fontsize=14)
ax1_abs.set_ylabel('Number of Companies')
ax1_abs.tick_params(axis='x', rotation=45)
ax1_abs.grid(axis='y', alpha=0.3)
ax1_abs.legend(title='Number of 1s', loc='upper right', fontsize='small')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
image1_path = 'Image1_2022_3fields.png'
plt.savefig(image1_path, dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# IMAGE 2: 2023 - PERCENTAGE AND ABSOLUTE VALUES
# =============================================================================
fig2, (ax2_pct, ax2_abs) = plt.subplots(1, 2, figsize=(24, 8))
fig2.suptitle('2023 - Distribution of Companies with at least one "1" (3 Fields)', fontsize=16, fontweight='bold')

# Prepare 2023 data
year_2023_data = detailed_breakdown[detailed_breakdown['year'] == '2023'].set_index('ateco_letter')
year_2023_data = year_2023_data.reindex(sector_order)

# PERCENTAGE PLOT
bottom_pct = np.zeros(len(sector_order))
for count, color in zip(range(1, 4), bar_colors):  # 1 to 3
    pct_col = f'pct_{count}'
    if pct_col in year_2023_data.columns:
        values = year_2023_data[pct_col].values
        bars = ax2_pct.bar(year_2023_data.index, values, bottom=bottom_pct,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_pct += values

        # Add percentage labels for values > 3%
        for i, val in enumerate(values):
            if val > 3:
                ax2_pct.text(i, bottom_pct[i] - val / 2, f'{val:.0f}%',
                             ha='center', va='center', fontsize=7, fontweight='bold')

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
ax2_pct.legend(title='Number of 1s', loc='upper right', fontsize='small')

# ABSOLUTE PLOT
bottom_abs = np.zeros(len(sector_order))
for count, color in zip(range(1, 4), bar_colors):  # 1 to 3
    if count in year_2023_data.columns:
        values = year_2023_data[count].values
        bars = ax2_abs.bar(year_2023_data.index, values, bottom=bottom_abs,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_abs += values

        # Add value labels for counts > 0
        for i, val in enumerate(values):
            if val > 0:
                ax2_abs.text(i, bottom_abs[i] - val / 2, f'{int(val)}',
                             ha='center', va='center', fontsize=7, fontweight='bold')

# Add total absolute line and labels
for i, total_abs in enumerate(bottom_abs):
    if total_abs > 0:
        ax2_abs.text(i, total_abs + 1, f'{int(total_abs)}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

ax2_abs.set_title('Absolute Counts', fontweight='bold', fontsize=14)
ax2_abs.set_ylabel('Number of Companies')
ax2_abs.tick_params(axis='x', rotation=45)
ax2_abs.grid(axis='y', alpha=0.3)
ax2_abs.legend(title='Number of 1s', loc='upper right', fontsize='small')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
image2_path = 'Image2_2023_3fields.png'
plt.savefig(image2_path, dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# IMAGE 3: 2024 - PERCENTAGE AND ABSOLUTE VALUES
# =============================================================================
fig3, (ax3_pct, ax3_abs) = plt.subplots(1, 2, figsize=(24, 8))
fig3.suptitle('2024 - Distribution of Companies with at least one "1" (3 Fields)', fontsize=16, fontweight='bold')

# Prepare 2024 data
year_2024_data = detailed_breakdown[detailed_breakdown['year'] == '2024'].set_index('ateco_letter')
year_2024_data = year_2024_data.reindex(sector_order)

# PERCENTAGE PLOT
bottom_pct = np.zeros(len(sector_order))
for count, color in zip(range(1, 4), bar_colors):  # 1 to 3
    pct_col = f'pct_{count}'
    if pct_col in year_2024_data.columns:
        values = year_2024_data[pct_col].values
        bars = ax3_pct.bar(year_2024_data.index, values, bottom=bottom_pct,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_pct += values

        # Add percentage labels for values > 3%
        for i, val in enumerate(values):
            if val > 3:
                ax3_pct.text(i, bottom_pct[i] - val / 2, f'{val:.0f}%',
                             ha='center', va='center', fontsize=7, fontweight='bold')

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
ax3_pct.legend(title='Number of 1s', loc='upper right', fontsize='small')

# ABSOLUTE PLOT
bottom_abs = np.zeros(len(sector_order))
for count, color in zip(range(1, 4), bar_colors):  # 1 to 3
    if count in year_2024_data.columns:
        values = year_2024_data[count].values
        bars = ax3_abs.bar(year_2024_data.index, values, bottom=bottom_abs,
                           color=color, label=f'{count} ones', alpha=0.8)
        bottom_abs += values

        # Add value labels for counts > 0
        for i, val in enumerate(values):
            if val > 0:
                ax3_abs.text(i, bottom_abs[i] - val / 2, f'{int(val)}',
                             ha='center', va='center', fontsize=7, fontweight='bold')

# Add total absolute line and labels
for i, total_abs in enumerate(bottom_abs):
    if total_abs > 0:
        ax3_abs.text(i, total_abs + 1, f'{int(total_abs)}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

ax3_abs.set_title('Absolute Counts', fontweight='bold', fontsize=14)
ax3_abs.set_ylabel('Number of Companies')
ax3_abs.tick_params(axis='x', rotation=45)
ax3_abs.grid(axis='y', alpha=0.3)
ax3_abs.legend(title='Number of 1s', loc='upper right', fontsize='small')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
image3_path = 'Image3_2024_3fields.png'
plt.savefig(image3_path, dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# PERFORM ANALYSIS
# =============================================================================

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
            pct_3 = row.loc['pct_3'] if 'pct_3' in row.index else 0  # Changed from pct_5 to pct_3

            # Calculate sum of companies with at least one 1
            total_with_ones = 0
            for count in range(1, 4):  # 1 to 3
                if count in row.index:
                    total_with_ones += row.loc[count]

            # Calculate sum of companies with 3 ones (excellence - all fields)
            excellence_count = row.loc[3] if 3 in row.index else 0

            sector_data[f'Pct_1plus_{year}'] = 100 - pct_0  # Percentage with at least one 1
            sector_data[f'Abs_1plus_{year}'] = total_with_ones  # Absolute with at least one 1
            sector_data[f'Pct_3ones_{year}'] = pct_3  # Percentage with 3 ones (full excellence)
            sector_data[f'Abs_3ones_{year}'] = excellence_count  # Absolute with 3 ones

    performance_analysis = pd.concat([performance_analysis, pd.DataFrame([sector_data])], ignore_index=True)

# Calculate trends
performance_analysis['Trend_1plus_Pct'] = performance_analysis['Pct_1plus_2024'] - performance_analysis['Pct_1plus_2022']
performance_analysis['Trend_3ones_Pct'] = performance_analysis['Pct_3ones_2024'] - performance_analysis['Pct_3ones_2022']
performance_analysis['Avg_Pct_1plus'] = performance_analysis[['Pct_1plus_2022', 'Pct_1plus_2023', 'Pct_1plus_2024']].mean(axis=1)

# Classify sectors by performance
high_performers = performance_analysis[performance_analysis['Avg_Pct_1plus'] > 60]
medium_performers = performance_analysis[(performance_analysis['Avg_Pct_1plus'] >= 30) &
                                         (performance_analysis['Avg_Pct_1plus'] <= 60)]
low_performers = performance_analysis[performance_analysis['Avg_Pct_1plus'] < 30]

# Biggest improvers and decliners
biggest_improvers = performance_analysis.nlargest(5, 'Trend_1plus_Pct')
biggest_decliners = performance_analysis.nsmallest(5, 'Trend_1plus_Pct')

# Excellence leaders (3 ones = all fields completed)
excellence_leaders = performance_analysis.nlargest(5, 'Pct_3ones_2024')

# Large sectors with good performance
large_engaged = performance_analysis[
    (performance_analysis['Total_Companies'] > 20) &
    (performance_analysis['Avg_Pct_1plus'] > 40)
].sort_values('Total_Companies', ascending=False)

# Small sectors with exceptional performance
small_excellent = performance_analysis[
    (performance_analysis['Total_Companies'] <= 5) &
    (performance_analysis['Avg_Pct_1plus'] > 70)
].sort_values('Avg_Pct_1plus', ascending=False)

# Distribution analysis for 2024
total_companies_2024 = len(df)
year_2024_summary = detailed_breakdown[detailed_breakdown['year'] == '2024'].groupby('year').sum()
distribution_data = []
if not year_2024_summary.empty:
    for count in range(0, 4):  # 0 to 3
        if count in year_2024_summary.columns:
            num_companies = year_2024_summary.loc['2024', count]
            percentage = (num_companies / total_companies_2024 * 100)
            distribution_data.append((count, int(num_companies), f"{percentage:.1f}%"))

# Overall engagement trends
overall_engagement_2022 = df['count_ones_2022'].apply(lambda x: 1 if x > 0 else 0).mean() * 100
overall_engagement_2023 = df['count_ones_2023'].apply(lambda x: 1 if x > 0 else 0).mean() * 100
overall_engagement_2024 = df['count_ones_2024'].apply(lambda x: 1 if x > 0 else 0).mean() * 100
overall_trend = overall_engagement_2024 - overall_engagement_2022

# Average number of '1s' per engaged company
avg_ones_data = {}
for year in ['2022', '2023', '2024']:
    engaged_companies = df[df[f'count_ones_{year}'] > 0]
    if len(engaged_companies) > 0:
        avg_ones = engaged_companies[f'count_ones_{year}'].mean()
        avg_ones_data[year] = avg_ones

# Prepare summary table
summary_2024 = performance_analysis[['Sector', 'Total_Companies', 'Pct_1plus_2024', 'Pct_3ones_2024', 'Trend_1plus_Pct']].copy()
summary_2024 = summary_2024.sort_values('Pct_1plus_2024', ascending=False)
summary_2024.columns = ['Sector', 'Total Cos', '2024: % ≥1', '2024: % 3', 'Trend 22-24']
summary_table_data = summary_2024.round(1).values.tolist()
summary_table_headers = ['Sector', 'Total Cos', '2024: % ≥1', '2024: % 3', 'Trend 22-24']

# =============================================================================
# CREATE PDF REPORT FUNCTION - ADJUSTED FOR 3 FIELDS
# =============================================================================

def create_pdf_report_3fields():
    # Import colors locally to avoid conflicts
    from reportlab.lib import colors

    # Create PDF document
    pdf_filename = f"Analysis_Report_3fields_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=landscape(letter))

    # Get styles
    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        alignment=TA_CENTER,
        spaceAfter=30
    )

    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceBefore=20,
        spaceAfter=10
    )

    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#3498db'),
        spaceBefore=15,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leading=14
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=4,
        leading=12
    )

    # Create story (content) list
    story = []

    # Title page
    story.append(Paragraph("ATECO Sector Engagement Analysis Report (3 Fields)", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Analysis Date: {datetime.datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph("Dataset: Starting_Dataset.csv.csv (3 fields analysis)", styles['Normal']))
    story.append(Spacer(1, 40))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading1_style))
    story.append(Paragraph(f"This report analyzes engagement patterns across ATECO sectors from 2022 to 2024, "
                           f"tracking binary field completion across 3 fields per company. "
                           f"The analysis covers {len(df)} companies across {len(sector_order)} ATECO sectors.",
                           normal_style))
    story.append(Spacer(1, 10))

    # Key findings table
    key_data = [
        ['Metric', 'Value'],
        ['Total Companies Analyzed', f"{len(df):,}"],
        ['Total ATECO Sectors', f"{len(sector_order)}"],
        ['Total Fields per Company', '3'],
        ['Overall Engagement (2024)', f"{overall_engagement_2024:.1f}%"],
        ['Overall Engagement Trend (2022-2024)', f"{overall_trend:+.1f}%"],
        ['High Performing Sectors', f"{len(high_performers)}"],
        ['Medium Performing Sectors', f"{len(medium_performers)}"],
        ['Low Performing Sectors', f"{len(low_performers)}"]
    ]

    key_table = Table(key_data, colWidths=[200, 100])
    key_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(key_table)
    story.append(PageBreak())

    # Section 1: Visualizations
    story.append(Paragraph("1. Visual Analysis of Engagement Patterns (3 Fields)", heading1_style))

    # 2022 Image
    story.append(Paragraph("2022 - Distribution of Companies with at least one '1'", heading2_style))
    try:
        story.append(Image(image1_path, width=9 * inch, height=3.5 * inch))
    except:
        story.append(Paragraph("Image not available", normal_style))
    story.append(Spacer(1, 20))

    # 2023 Image
    story.append(Paragraph("2023 - Distribution of Companies with at least one '1'", heading2_style))
    try:
        story.append(Image(image2_path, width=9 * inch, height=3.5 * inch))
    except:
        story.append(Paragraph("Image not available", normal_style))
    story.append(Spacer(1, 20))

    # 2024 Image
    story.append(Paragraph("2024 - Distribution of Companies with at least one '1'", heading2_style))
    try:
        story.append(Image(image3_path, width=9 * inch, height=3.5 * inch))
    except:
        story.append(Paragraph("Image not available", normal_style))
    story.append(PageBreak())

    # Section 2: Sector Performance Classification
    story.append(Paragraph("2. Sector Performance Classification", heading1_style))
    story.append(Paragraph(f"High Performers (>60% average engagement): {len(high_performers)} sectors", heading2_style))

    for _, row in high_performers.sort_values('Avg_Pct_1plus', ascending=False).iterrows():
        story.append(Paragraph(
            f"• {row['Sector']}: {row['Avg_Pct_1plus']:.1f}% avg engagement, {row['Total_Companies']} companies",
            bullet_style))

    story.append(Paragraph(f"Medium Performers (30-60% average engagement): {len(medium_performers)} sectors", heading2_style))

    for _, row in medium_performers.sort_values('Avg_Pct_1plus', ascending=False).iterrows():
        story.append(Paragraph(
            f"• {row['Sector']}: {row['Avg_Pct_1plus']:.1f}% avg engagement, {row['Total_Companies']} companies",
            bullet_style))

    story.append(Paragraph(f"Low Performers (<30% average engagement): {len(low_performers)} sectors", heading2_style))

    for _, row in low_performers.sort_values('Avg_Pct_1plus', ascending=False).iterrows():
        story.append(Paragraph(
            f"• {row['Sector']}: {row['Avg_Pct_1plus']:.1f}% avg engagement, {row['Total_Companies']} companies",
            bullet_style))

    story.append(PageBreak())

    # Section 3: Trend Analysis
    story.append(Paragraph("3. Trend Analysis (2022-2024)", heading1_style))

    story.append(Paragraph("Biggest Improvements in Engagement:", heading2_style))
    for _, row in biggest_improvers.iterrows():
        story.append(Paragraph(f"• {row['Sector']}: +{row['Trend_1plus_Pct']:.1f}% "
                               f"({row['Pct_1plus_2022']:.1f}% → {row['Pct_1plus_2024']:.1f}%)", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Biggest Declines in Engagement:", heading2_style))
    for _, row in biggest_decliners.iterrows():
        if row['Trend_1plus_Pct'] < 0:
            story.append(Paragraph(f"• {row['Sector']}: {row['Trend_1plus_Pct']:.1f}% "
                                   f"({row['Pct_1plus_2022']:.1f}% → {row['Pct_1plus_2024']:.1f}%)", bullet_style))

    # Section 4: Excellence Analysis
    story.append(Paragraph("4. Excellence Analysis (Companies with all 3 '1s')", heading1_style))
    story.append(Paragraph("Sectors with Highest Excellence (3 '1s') in 2024:", heading2_style))

    for _, row in excellence_leaders.iterrows():
        if row['Pct_3ones_2024'] > 0:
            story.append(
                Paragraph(f"• {row['Sector']}: {row['Pct_3ones_2024']:.1f}% ({row['Abs_3ones_2024']} companies)",
                          bullet_style))

    story.append(PageBreak())

    # Section 5: Sector Size vs Engagement
    story.append(Paragraph("5. Sector Size vs Engagement Analysis", heading1_style))

    story.append(Paragraph("Large Sectors (>20 companies) with Good Engagement (>40%):", heading2_style))
    if len(large_engaged) > 0:
        for _, row in large_engaged.iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row['Total_Companies']} companies, {row['Avg_Pct_1plus']:.1f}% avg engagement",
                bullet_style))
    else:
        story.append(Paragraph("No large sectors meeting the criteria", normal_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Small Sectors (≤5 companies) with Exceptional Performance (>70%):", heading2_style))
    if len(small_excellent) > 0:
        for _, row in small_excellent.iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row['Total_Companies']} companies, {row['Avg_Pct_1plus']:.1f}% avg engagement",
                bullet_style))
    else:
        story.append(Paragraph("No small sectors meeting the criteria", normal_style))

    # Section 6: Distribution Analysis
    story.append(Paragraph("6. Distribution of Number of '1s' Across All Companies (2024)", heading1_style))
    story.append(Paragraph(f"Total companies analyzed: {total_companies_2024:,}", normal_style))

    # Create distribution table
    dist_headers = ['Number of 1s', 'Companies', 'Percentage']
    dist_data = [dist_headers] + distribution_data

    dist_table = Table(dist_data, colWidths=[150, 100, 100])
    dist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    story.append(dist_table)
    story.append(PageBreak())

    # Section 7: Performance Summary Table
    story.append(Paragraph("7. Performance Summary Table (2024)", heading1_style))
    story.append(Paragraph("All sectors sorted by percentage of companies with at least one '1' in 2024", normal_style))

    # Create summary table
    summary_data = [summary_table_headers] + summary_table_data

    summary_table = Table(summary_data, colWidths=[80, 80, 100, 80, 80])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 20))

    # Section 8: Overall Engagement Trends
    story.append(Paragraph("8. Overall Engagement Trends (All Sectors Combined)", heading1_style))

    overall_data = [
        ['Year', 'Percentage with at least one "1"', 'Trend'],
        ['2022', f"{overall_engagement_2022:.1f}%", '-'],
        ['2023', f"{overall_engagement_2023:.1f}%", f"{overall_engagement_2023 - overall_engagement_2022:+.1f}%"],
        ['2024', f"{overall_engagement_2024:.1f}%", f"{overall_engagement_2024 - overall_engagement_2023:+.1f}%"],
        ['Overall Trend (2022-2024)', f"{overall_engagement_2024:.1f}%", f"{overall_trend:+.1f}%"]
    ]

    overall_table = Table(overall_data, colWidths=[120, 150, 100])
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    story.append(overall_table)
    story.append(Spacer(1, 20))

    # Average number of '1s' per engaged company
    story.append(Paragraph("Average Number of '1s' per Engaged Company:", heading2_style))
    for year, avg_ones in avg_ones_data.items():
        story.append(Paragraph(f"• {year}: {avg_ones:.2f} '1s' per engaged company", bullet_style))

    # Conclusions
    story.append(PageBreak())
    story.append(Paragraph("Conclusions and Recommendations", heading1_style))

    conclusions = [
        "1. Focus on low-performing sectors to improve overall engagement rates across all 3 fields.",
        "2. Investigate sectors showing declining trends to understand barriers to complete engagement.",
        "3. Recognize and learn from high-performing sectors to replicate successful practices.",
        "4. Monitor small sectors with exceptional performance as potential benchmarks.",
        "5. Use the detailed distribution analysis to target specific engagement levels (partial vs full completion).",
        "6. Companies with 3 '1s' represent full compliance - track these for best practices.",
        "7. Continue tracking trends to measure the impact of engagement initiatives across all 3 fields."
    ]

    for conclusion in conclusions:
        story.append(Paragraph(conclusion, bullet_style))

    # Footer
    story.append(Spacer(1, 40))
    story.append(Paragraph(f"Report generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
                           ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER,
                                          textColor=colors.gray)))

    # Build the PDF
    doc.build(story)

    return pdf_filename

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        print(f"✓ Dataset loaded: {len(df)} companies across {len(sector_order)} ATECO sectors")
        print(f"✓ ATECO sectors found: {', '.join(sorted(sector_order))}")
        print(f"✓ Years analyzed: 2022, 2023, 2024")
        print(f"✓ Fields analyzed: Field1 to Field3 (3 fields total)")

        # Create and save visualizations
        print(f"✓ Creating visualizations...")

        # Generate PDF report
        print(f"✓ Generating PDF report...")
        pdf_file = create_pdf_report_3fields()

        print(f"\n✓ Analysis complete!")
        print(f"✓ PDF report successfully created: {pdf_file}")
        print(f"✓ Images saved as: {image1_path}, {image2_path}, {image3_path}")
        print(f"\nSummary Statistics:")
        print(f"  - Overall engagement (2024): {overall_engagement_2024:.1f}%")
        print(f"  - High performing sectors: {len(high_performers)}")
        print(f"  - Medium performing sectors: {len(medium_performers)}")
        print(f"  - Low performing sectors: {len(low_performers)}")
        print(f"  - Maximum possible '1s' per company: 3")

    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")