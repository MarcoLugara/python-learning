import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import datetime
import os
import colorsys


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
        #These were the black ones
        'QUESTIONARI': 'Field9',
        'FOCUS GROUP': 'Field10',
        'WORKSHOP': 'Field11',
        'GROUP MEETING': 'Field12',
        'INCONTRI PERIODICI': 'Field13',
        # These were the red ones
        'INTRANET AZIENDALE ': 'Field14',
        'NEWSLETTERS': 'Field15',
        'COMUNICATI STAMPA': 'Field16'
        #These were the blue ones

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


'''
df = comprehensive_database_reshaping(
    "Database Ufficiale.csv",
    "ATECO_codes.csv"
)
'''

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

# Load the data from your provided CSV
df = pd.read_csv('Tidier_Dataset.csv')

# Extract ATECO letter from the first column
df['ateco_letter'] = df['ateco'].str[0]

# Define field groups with their color schemes
field_groups = {
    'Red_Group': {
        'fields': [f'Field{i}' for i in range(1, 9)],  # Fields 1-8
        'color_base': '#ff6b6b',  # Red base
        'name': 'Fields 1-8',
        'max_fields': 8
    },
    'Blue_Group': {
        'fields': [f'Field{i}' for i in range(9, 14)],  # Fields 9-13
        'color_base': '#4d96ff',  # Blue base
        'name': 'Fields 9-13',
        'max_fields': 5
    },
    'Green_Group': {
        'fields': [f'Field{i}' for i in range(14, 17)],  # Fields 14-16
        'color_base': '#6bcf7f',  # Green base
        'name': 'Fields 14-16',
        'max_fields': 3
    }
}

# Calculate sum of ones for each group and year
for group_name, group_info in field_groups.items():
    for year in ['2022', '2023', '2024']:
        field_columns = [f'{field}_{year}' for field in group_info['fields']]
        df[f'{group_name}_sum_{year}'] = df[field_columns].sum(axis=1)

# Calculate total sum (all 16 fields) for each year
for year in ['2022', '2023', '2024']:
    all_fields = [col for col in df.columns if f'_{year}' in col and 'Field' in col]
    df[f'total_sum_{year}'] = df[all_fields].sum(axis=1)


# =============================================================================
# COLOR FUNCTIONS
# =============================================================================

def get_color_gradient(base_color, num_shades, reverse=False):
    """Generate a color gradient from light to dark or vice versa."""
    # Convert hex to RGB
    base_color = base_color.lstrip('#')
    rgb = tuple(int(base_color[i:i + 2], 16) for i in (0, 2, 4))

    # Convert RGB to HSL
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Generate shades
    shades = []
    if reverse:
        light_values = np.linspace(0.8, 0.3, num_shades)  # Dark to light
    else:
        light_values = np.linspace(0.3, 0.8, num_shades)  # Light to dark

    for lightness in light_values:
        # Keep hue and saturation, adjust lightness
        r, g, b = colorsys.hls_to_rgb(h, lightness, s)
        # Convert back to hex
        hex_color = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
        shades.append(hex_color)

    return shades


# Generate color gradients for each group
group_colors = {}
for group_name, group_info in field_groups.items():
    # Generate colors for each possible value (0 to max_fields)
    num_shades = group_info['max_fields'] + 1
    group_colors[group_name] = get_color_gradient(group_info['color_base'], num_shades, reverse=True)

# =============================================================================
# PREPARE DATA FOR VISUALIZATION
# =============================================================================

# Create detailed breakdown by ATECO letter and sum values for each group
detailed_breakdown = pd.DataFrame()

for year in ['2022', '2023', '2024']:
    year_data_list = []

    for sector in df['ateco_letter'].unique():
        sector_data = df[df['ateco_letter'] == sector]

        # For each group, count companies with each sum value
        sector_counts = {'ateco_letter': sector, 'year': year}

        for group_name, group_info in field_groups.items():
            sum_col = f'{group_name}_sum_{year}'
            for value in range(0, group_info['max_fields'] + 1):
                count = len(sector_data[sector_data[sum_col] == value])
                sector_counts[f'{group_name}_{value}'] = count

        # Total companies in sector
        sector_counts['total_companies'] = len(sector_data)

        year_data_list.append(sector_counts)

    year_df = pd.DataFrame(year_data_list)
    detailed_breakdown = pd.concat([detailed_breakdown, year_df], ignore_index=True)

# Fill NaN with 0
detailed_breakdown = detailed_breakdown.fillna(0)

# Get sector order by total companies (for consistent ordering)
sector_order = df['ateco_letter'].value_counts().index


# =============================================================================
# CREATE UNIFIED VISUALIZATIONS
# =============================================================================

def create_unified_bar_chart(year_data, year, sector_order):
    """Create a unified bar chart with stacked segments for a specific year."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Set up bar positions and width
    x_positions = np.arange(len(sector_order))
    bar_width = 0.6

    # We'll stack by groups, then by values within each group
    # Create a list to store all legend handles and labels
    legend_handles = []
    legend_labels = []

    # First, for each sector, we need to calculate the order of stacking
    # We'll stack: Red (0-8), Blue (0-5), Green (0-3)

    # Prepare data structure: sector -> group -> value -> count
    sector_data = {}
    for sector in sector_order:
        sector_row = year_data[year_data['ateco_letter'] == sector]
        if not sector_row.empty:
            sector_dict = {}
            for group_name in field_groups.keys():
                group_dict = {}
                max_fields = field_groups[group_name]['max_fields']
                for value in range(max_fields, -1, -1):  # From highest to lowest for stacking
                    count = sector_row.iloc[0][f'{group_name}_{value}']
                    group_dict[value] = count
                sector_dict[group_name] = group_dict
            sector_data[sector] = sector_dict

    # Create the stacked bars
    # We need to stack: first all Red values, then Blue, then Green
    group_order = ['Red_Group', 'Blue_Group', 'Green_Group']

    # Initialize bottoms for each bar
    bottoms = np.zeros(len(sector_order))

    # Keep track of which colors we've added to legend
    legend_added = {}

    # Stack in order: Red (0-8), Blue (0-5), Green (0-3)
    # For visual clarity, we'll stack from lowest to highest value within each group
    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Stack from 0 to max_fields (lightest to darkest)
        for value in range(0, max_fields + 1):
            # Get counts for this value across all sectors
            counts = []
            for i, sector in enumerate(sector_order):
                if sector in sector_data:
                    counts.append(sector_data[sector][group_name].get(value, 0))
                else:
                    counts.append(0)

            if sum(counts) > 0:  # Only add if there are any
                color_idx = value  # 0 = lightest, max_fields = darkest
                color = colors_for_group[color_idx]

                bars = ax.bar(x_positions, counts, bottom=bottoms,
                              width=bar_width, color=color, alpha=0.8)
                bottoms += counts

                # Add to legend if not already added
                legend_key = f"{group_name}_{value}"
                if legend_key not in legend_added:
                    legend_handles.append(Patch(facecolor=color, edgecolor='black', alpha=0.8))
                    legend_labels.append(f"{group_info['name']}: {value} fields")
                    legend_added[legend_key] = True

    # Customize the plot
    ax.set_title(f'{year} - Field Completion Distribution by ATECO Sector',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ATECO Sector', fontsize=12)
    ax.set_ylabel('Number of Companies', fontsize=12)

    # Set x-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sector_order, rotation=45, ha='right', fontsize=10)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total count labels on top of bars
    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(i, total + 0.5, f'{int(total)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Create a comprehensive legend
    # Group legend items by field group
    red_items = []
    blue_items = []
    green_items = []

    for handle, label in zip(legend_handles, legend_labels):
        if 'Fields 1-8' in label:
            red_items.append((handle, label))
        elif 'Fields 9-13' in label:
            blue_items.append((handle, label))
        elif 'Fields 14-16' in label:
            green_items.append((handle, label))

    # Sort items by number of fields (ascending)
    red_items.sort(key=lambda x: int(x[1].split(': ')[1].split(' ')[0]))
    blue_items.sort(key=lambda x: int(x[1].split(': ')[1].split(' ')[0]))
    green_items.sort(key=lambda x: int(x[1].split(': ')[1].split(' ')[0]))

    # Create grouped legend
    all_handles = []
    all_labels = []

    # Add group headers
    from matplotlib.lines import Line2D
    group_headers = [
        Line2D([0], [0], color='white', label='Red Group: Fields 1-8'),
        *[handle for handle, label in red_items],
        Line2D([0], [0], color='white', label=' '),  # Spacer
        Line2D([0], [0], color='white', label='Blue Group: Fields 9-13'),
        *[handle for handle, label in blue_items],
        Line2D([0], [0], color='white', label=' '),  # Spacer
        Line2D([0], [0], color='white', label='Green Group: Fields 14-16'),
        *[handle for handle, label in green_items]
    ]

    # Create a separate legend for group headers
    header_labels = ['Red Group: Fields 1-8'] + [label for _, label in red_items] + \
                    [' '] + ['Blue Group: Fields 9-13'] + [label for _, label in blue_items] + \
                    [' '] + ['Green Group: Fields 14-16'] + [label for _, label in green_items]

    # Adjust layout to accommodate legend
    plt.tight_layout(rect=[0, 0, 0.75, 0.95])

    # Create legend outside the plot
    legend = ax.legend(group_headers, header_labels,
                       title='Field Completion Levels',
                       loc='center left',
                       bbox_to_anchor=(1.02, 0.5),
                       fontsize=9,
                       title_fontsize=10,
                       frameon=True,
                       fancybox=True,
                       shadow=True)

    # Add explanation text
    explanation_text = (
        "Color Intensity: Darker shades indicate higher number of completed fields within each group\n"
        "Bar Height: Total number of companies in each sector\n"
        "Stacking: Shows distribution of field completion levels"
    )

    plt.figtext(0.5, 0.01, explanation_text, ha='center', fontsize=9,
                style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig


# Create visualizations for each year
image_paths = []
for year in ['2022', '2023', '2024']:
    year_data = detailed_breakdown[detailed_breakdown['year'] == year]
    fig = create_unified_bar_chart(year_data, year, sector_order)

    image_path = f'Image_{year}_Unified_Stacked.png'
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    image_paths.append(image_path)
    plt.close()

# =============================================================================
# PERFORM ANALYSIS (same as before)
# =============================================================================

# Calculate performance metrics
performance_analysis = pd.DataFrame()

for sector in sector_order:
    sector_data = {'Sector': sector, 'Total_Companies': len(df[df['ateco_letter'] == sector])}

    for year in ['2022', '2023', '2024']:
        year_df = detailed_breakdown[(detailed_breakdown['year'] == year) &
                                     (detailed_breakdown['ateco_letter'] == sector)]

        if not year_df.empty:
            row = year_df.iloc[0]

            # Calculate engagement metrics for each group
            for group_name, group_info in field_groups.items():
                total_possible = group_info['max_fields'] * sector_data['Total_Companies']
                actual_sum = 0
                for value in range(group_info['max_fields'] + 1):
                    actual_sum += value * row[f'{group_name}_{value}']

                if total_possible > 0:
                    completion_rate = (actual_sum / total_possible) * 100
                else:
                    completion_rate = 0

                sector_data[f'{group_name}_completion_{year}'] = completion_rate
                sector_data[f'{group_name}_avg_{year}'] = actual_sum / sector_data['Total_Companies'] if sector_data[
                                                                                                             'Total_Companies'] > 0 else 0

            # Overall completion (all 16 fields)
            total_fields_completed = 0
            for group_name in field_groups.keys():
                for value in range(field_groups[group_name]['max_fields'] + 1):
                    total_fields_completed += value * row[f'{group_name}_{value}']

            total_possible_all = 16 * sector_data['Total_Companies']
            if total_possible_all > 0:
                overall_completion = (total_fields_completed / total_possible_all) * 100
            else:
                overall_completion = 0

            sector_data[f'overall_completion_{year}'] = overall_completion
            sector_data[f'avg_fields_per_company_{year}'] = total_fields_completed / sector_data['Total_Companies'] if \
            sector_data['Total_Companies'] > 0 else 0

    performance_analysis = pd.concat([performance_analysis, pd.DataFrame([sector_data])], ignore_index=True)

# Calculate trends
performance_analysis['trend_overall_22_24'] = performance_analysis['overall_completion_2024'] - performance_analysis[
    'overall_completion_2022']

# Calculate average completion rates
for group_name in field_groups.keys():
    performance_analysis[f'avg_{group_name}_completion'] = performance_analysis[[
        f'{group_name}_completion_2022',
        f'{group_name}_completion_2023',
        f'{group_name}_completion_2024'
    ]].mean(axis=1)

performance_analysis['avg_overall_completion'] = performance_analysis[[
    'overall_completion_2022', 'overall_completion_2023', 'overall_completion_2024'
]].mean(axis=1)

# Classify sectors by overall performance
high_performers = performance_analysis[performance_analysis['avg_overall_completion'] > 40]
medium_performers = performance_analysis[(performance_analysis['avg_overall_completion'] >= 20) &
                                         (performance_analysis['avg_overall_completion'] <= 40)]
low_performers = performance_analysis[performance_analysis['avg_overall_completion'] < 20]

# Find biggest improvers and decliners
biggest_improvers = performance_analysis.nlargest(5, 'trend_overall_22_24')
biggest_decliners = performance_analysis.nsmallest(5, 'trend_overall_22_24')

# Find sectors with highest average completion in each group
group_leaders = {}
for group_name in field_groups.keys():
    group_leaders[group_name] = performance_analysis.nlargest(5, f'avg_{group_name}_completion')

# Overall statistics
overall_completion_2022 = performance_analysis['overall_completion_2022'].mean()
overall_completion_2023 = performance_analysis['overall_completion_2023'].mean()
overall_completion_2024 = performance_analysis['overall_completion_2024'].mean()
overall_trend = overall_completion_2024 - overall_completion_2022

# Create summary table
summary_data = performance_analysis[['Sector', 'Total_Companies',
                                     'overall_completion_2024',
                                     'avg_fields_per_company_2024',
                                     'trend_overall_22_24']].copy()
summary_data = summary_data.sort_values('overall_completion_2024', ascending=False)
summary_data.columns = ['Sector', 'Total Cos', '2024: Overall %', '2024: Avg Fields', 'Trend 22-24']
summary_table_data = summary_data.round(1).values.tolist()
summary_table_headers = ['Sector', 'Total Cos', '2024: Overall %', '2024: Avg Fields', 'Trend 22-24']


# =============================================================================
# CREATE PDF REPORT (simplified version)
# =============================================================================

def create_pdf_report():
    from reportlab.lib import colors

    # Create PDF document
    pdf_filename = f"Analysis_Report_Unified_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
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
    story.append(Paragraph("ATECO Sector Analysis - Unified Field Groups Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Analysis Date: {datetime.datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph("Dataset: Tidier_Dataset.csv (16 fields grouped by color)", styles['Normal']))
    story.append(Spacer(1, 20))

    # Field groups description
    story.append(Paragraph("Field Groups:", heading2_style))
    story.append(Paragraph(
        "• <font color='#ff6b6b'>Red Group (Fields 1-8):</font> 8 fields - Gradient from light to dark red indicates 0-8 completed fields",
        bullet_style))
    story.append(Paragraph(
        "• <font color='#4d96ff'>Blue Group (Fields 9-13):</font> 5 fields - Gradient from light to dark blue indicates 0-5 completed fields",
        bullet_style))
    story.append(Paragraph(
        "• <font color='#6bcf7f'>Green Group (Fields 14-16):</font> 3 fields - Gradient from light to dark green indicates 0-3 completed fields",
        bullet_style))

    story.append(Spacer(1, 20))

    # Key findings table
    key_data = [
        ['Metric', 'Value'],
        ['Total Companies Analyzed', f"{len(df):,}"],
        ['Total ATECO Sectors', f"{len(sector_order)}"],
        ['Total Fields per Company', '16'],
        ['Field Groups', '3 (Red, Blue, Green)'],
        ['Overall Completion (2024)', f"{overall_completion_2024:.1f}%"],
        ['Overall Trend (2022-2024)', f"{overall_trend:+.1f}%"],
        ['High Performing Sectors (>40%)', f"{len(high_performers)}"],
        ['Medium Performing Sectors (20-40%)', f"{len(medium_performers)}"],
        ['Low Performing Sectors (<20%)', f"{len(low_performers)}"]
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
    story.append(Paragraph("1. Unified Stacked Bar Charts by Year", heading1_style))
    story.append(Paragraph(
        "Each bar represents one ATECO sector. Colors show distribution of field completion within each group.",
        normal_style))

    for year, image_path in zip(['2022', '2023', '2024'], image_paths):
        story.append(Paragraph(f"{year} - Field Completion Distribution", heading2_style))
        try:
            story.append(Image(image_path, width=11 * inch, height=6 * inch))
        except:
            story.append(Paragraph("Image not available", normal_style))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # Section 2: Sector Performance Classification
    story.append(Paragraph("2. Sector Performance Classification", heading1_style))
    story.append(
        Paragraph(f"High Performers (>40% overall completion): {len(high_performers)} sectors", heading2_style))

    for _, row in high_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
        story.append(Paragraph(
            f"• {row['Sector']}: {row['avg_overall_completion']:.1f}% avg completion, "
            f"{row['Total_Companies']} companies",
            bullet_style))

    story.append(
        Paragraph(f"Medium Performers (20-40% overall completion): {len(medium_performers)} sectors", heading2_style))

    for _, row in medium_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
        story.append(Paragraph(
            f"• {row['Sector']}: {row['avg_overall_completion']:.1f}% avg completion, "
            f"{row['Total_Companies']} companies",
            bullet_style))

    story.append(Paragraph(f"Low Performers (<20% overall completion): {len(low_performers)} sectors", heading2_style))

    for _, row in low_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
        story.append(Paragraph(
            f"• {row['Sector']}: {row['avg_overall_completion']:.1f}% avg completion, "
            f"{row['Total_Companies']} companies",
            bullet_style))

    story.append(PageBreak())

    # Section 3: Performance Summary Table
    story.append(Paragraph("3. Performance Summary Table (2024)", heading1_style))
    story.append(Paragraph("All sectors sorted by overall completion percentage in 2024", normal_style))

    # Create summary table
    summary_data = [summary_table_headers] + summary_table_data

    summary_table = Table(summary_data, colWidths=[80, 80, 100, 100, 80])
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

    # Section 4: Overall Engagement Trends
    story.append(Paragraph("4. Overall Completion Trends (All Sectors Combined)", heading1_style))

    overall_data = [
        ['Year', 'Overall Completion %', 'Trend'],
        ['2022', f"{overall_completion_2022:.1f}%", '-'],
        ['2023', f"{overall_completion_2023:.1f}%",
         f"{overall_completion_2023 - overall_completion_2022:+.1f}%"],
        ['2024', f"{overall_completion_2024:.1f}%",
         f"{overall_completion_2024 - overall_completion_2023:+.1f}%"],
        ['Overall Trend (2022-2024)', f"{overall_completion_2024:.1f}%",
         f"{overall_trend:+.1f}%"]
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

    # Conclusions
    story.append(PageBreak())
    story.append(Paragraph("Conclusions and Recommendations", heading1_style))

    conclusions = [
        "1. The unified stacked bar charts provide a comprehensive view of field completion patterns across all three groups.",
        "2. Darker shades within each color group indicate higher numbers of completed fields.",
        "3. Focus improvement efforts on sectors with predominantly light colors (low completion).",
        "4. Sectors with balanced dark colors across all three groups represent best practices.",
        "5. Use the color gradients to quickly identify completion patterns within each field group.",
        "6. Monitor changes in color distribution year over year to track progress.",
        "7. Develop targeted strategies for each field group based on sector performance patterns."
    ]

    for conclusion in conclusions:
        story.append(Paragraph(conclusion, bullet_style))

    # Footer
    story.append(Spacer(1, 40))
    story.append(Paragraph(f"Report generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
                           ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                                          alignment=TA_CENTER, textColor=colors.gray)))

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
        print(f"✓ Fields analyzed: 16 fields grouped into 3 categories:")
        print(f"    - Red Group: Fields 1-8 (8 fields, gradient: light to dark red)")
        print(f"    - Blue Group: Fields 9-13 (5 fields, gradient: light to dark blue)")
        print(f"    - Green Group: Fields 14-16 (3 fields, gradient: light to dark green)")

        # Create and save visualizations
        print(f"✓ Creating unified visualizations...")

        # Generate PDF report
        print(f"✓ Generating PDF report...")
        pdf_file = create_pdf_report()

        print(f"\n✓ Analysis complete!")
        print(f"✓ PDF report successfully created: {pdf_file}")
        print(f"✓ Images saved as: {', '.join(image_paths)}")
        print(f"\nSummary Statistics:")
        print(f"  - Overall completion (2024): {overall_completion_2024:.1f}%")
        print(f"  - High performing sectors: {len(high_performers)}")
        print(f"  - Medium performing sectors: {len(medium_performers)}")
        print(f"  - Low performing sectors: {len(low_performers)}")
        print(f"  - Total fields per company: 16")
        print(f"\nColor Legend:")
        print(f"  - Red: Fields 1-8 (0-8 completed fields)")
        print(f"  - Blue: Fields 9-13 (0-5 completed fields)")
        print(f"  - Green: Fields 14-16 (0-3 completed fields)")
        print(f"  - Darker shades = more fields completed within each group")

    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        print(f"Error details: {traceback.format_exc()}")