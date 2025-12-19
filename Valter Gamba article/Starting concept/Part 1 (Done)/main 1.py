import numpy as np
import pandas as pd
import seaborn as sns
#import scipy.stats as stats
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
#from matplotlib.ticker import PercentFormatter
import csv
import re


#OUTDATED/EXECUTED PROCESSES
#Reshaping of the DF
def reshape_database(df1):
    """
    Complete database reshaping using a manual approach
    """
    # Rename columns for readability
    df2 = df1.rename(columns={
        'AZIENDE SUDDIVISE PER CODICE ATECO': 'Name',
        'ATECO 2007\ncodice': 'ATECO',
        'ATECO 2007\ndescrizione': 'ATECOx'}
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

    df.to_csv('Starting_Dataset.csv.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)

    """
    print(f"Original shape: {df3.shape}")  --> Original shape: (876, 7)
    print(f"Wide shape: {df.shape}") --> Wide shape: (291, 12) 291 companies analyzed
     Checks:
     name_list = df['Name'].tolist()
     print(f"Total entries: {len(name_list)}") --> Total entries: 291
    #print(df.columns)
    """
    return df

#3x3 table of piecharts for each distribution
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
    ##plt.show()

#Check of all 0s and all 1s and check of who did ESRS or SASB in 2024
def counts_and_percentages_and_law_check(df):
    # Computation of counts and percentages of the Industries with all 0s and 1s for each category throughout the 3 years
    #   and the computation of counts and percentage of Industries which presented either the SASB or the ESRS declaration in 2024
    standards = ['GRI', 'ESRS', 'SASB']
    for standard in standards:
        all0 = ((df[f'{standard}_2022'] == 0) & (df[f'{standard}_2023'] == 0) & (df[f'{standard}_2024'] == 0)).sum()
        all0_perc = round(all0 / len(df['Name']) * 100, 2)
        print(f'The number of industries which presented no {standard} istances between 2022 and 2024 are', all0,
              "namely", all0_perc, "%")

        all1 = ((df[f'{standard}_2022'] == 1) & (df[f'{standard}_2023'] == 1) & (df[f'{standard}_2024'] == 1)).sum()
        all1_perc = round(all1 / len(df['Name']) * 100, 2)
        print(f'The number of industries which presented all {standard} istances between 2022 and 2024 are', all1,
              "namely", all1_perc, "%")
        print(
            "This means that there are industries which haven't been consistent throughout the years, related to the pie chart")

    some_1s_by_law_in_2024 = ((df[f'{standards[1]}_2024'] == 1) | (df[f'{standards[2]}_2023'] == 1)).sum()
    some_1s_by_law_in_2024_perc = round(some_1s_by_law_in_2024 / len(df['Name']) * 100, 2)
    print("The number of industries which presented either the ESRS or the SASB instance in 2024 are",
          some_1s_by_law_in_2024, "namely", some_1s_by_law_in_2024_perc, "%")

def adding_new_Ateco_identifiers(df_Extra, df):
    df_Extra = pd.read_csv('../Part 2 (AA1000 only)/ATECO_codes.csv')
    df_Extra = df_Extra.rename(columns={
        'Codice Ateco': 'Codice',
        'Titolo Ateco 2007 aggiornamento 2022': 'Codice_desc'}
    )
    df_Extra = df_Extra[['Codice', 'Codice_desc']]

    """
    Basic Syntax for INDEXROW()
    for index, row in dataframe.iterrows():
        # index: the row index
        # row: a Series containing the row data
        # access row data using row['column_name']
    """
    # Getting the values from the ISTAT database related to the general ateco codes
    ateco = list()
    atecoX = list()
    for index, row in df_Extra.iterrows():
        if len(row['Codice']) == 2:
            ateco.append(row['Codice'])
            atecoX.append(row['Codice_desc'])
    # print(ateco)
    # print(atecoX)

    # Create mapping dictionary, which maps each n-th value in the first list to the n-th value in the second list
    mapping_dict = dict(zip(ateco, atecoX))

    """ ZIP FUNCTION
    The zip(iterable1, iterable2, iterable3, ...) function takes multiple iterables (lists, tuples, etc.) and returns an iterator of tuples where:
    The i-th tuple contains the i-th element from each of the input iterables
    It stops when the shortest input iterable is exhausted
    It's a perfect match if the length of all iterable1, iterable2, etc... is the same
    """

    # Creating the new 2 columns which have the first 2 characters of ATECO and the related description from the ISTAT DF
    df['Ateco'] = df['ATECO'].astype(str).str[:2]  # .str[:2] takes up to the 2nd character in the string
    df['AtecoX'] = df['Ateco'].map(mapping_dict)

    #Putting the new 2 columns before the ATECO ones
    new_cols = df[['Ateco', 'AtecoX']]
    df = df.drop(df.columns[12:], axis=1)  # cols.remove(thing) works only for one element, for a list. Drop works best for DF for rows and columns. SAY THE AXIS
    #print(df.columns)
    df = pd.concat([df.iloc[:, :1], new_cols, df.iloc[:, 1:]], axis=1)  # axis=0 concatenates rows, axis=1 concatenates columns
    #print(df.columns)

    # Updating the csv and excel
    df.to_csv('Starting_Dataset.csv.csv', index=False)
    df.to_excel('Tidier_Dataset.xlsx', index=False)
    # print(df.columns)

    # Sorting the DF list by Ateco code, in case it's needed somewhere
    df_sorted_by_Ateco = df.sort_values(['Ateco', 'ATECO'])
            #This will sort first by Ateco and then ONLY WHEN MATCHING, by ATECO
    return df, df_sorted_by_Ateco

def figure1(df):
    # Load and process the dataset
    data = df
    framework_columns = ['GRI_2022', 'ESRS_2022', 'SASB_2022', 'GRI_2023', 'ESRS_2023', 'SASB_2023', 'GRI_2024',
                         'ESRS_2024', 'SASB_2024']
    data['Total_Adoption'] = data[framework_columns].sum(axis=1)
    zero_adoption = data[data['Total_Adoption'] == 0].copy()

    # Extract 2-digit ATECO codes and count occurrences
    zero_adoption['Ateco_2digit'] = zero_adoption['Ateco'].astype(str).str[:2].fillna('na')
    ateco_counts = zero_adoption['Ateco_2digit'].value_counts().sort_index()

    # ATECOX descriptions for legend (unique 2-digit codes)
    ateco_descriptions = data[['Ateco', 'AtecoX']].drop_duplicates().set_index('Ateco')['AtecoX'].to_dict()
    ateco_labels = {code: ateco_descriptions.get(code, 'Not Available') for code in ateco_counts.index}

    # Create figure with a cool background
    plt.figure(figsize=(12, 6), facecolor='#f0f8ff')
    ax = plt.gca()
    ax.set_facecolor('#e6f0fa')

    # Bar plot with reversed gradient colors (red for high, green for low)
    x = np.arange(len(ateco_counts))
    # Normalize counts for color scaling (reversed: high = red, low = green)
    norm_counts = ateco_counts.values / ateco_counts.max()
    colors = plt.cm.RdYlGn_r(norm_counts)  # Reversed RdYlGn colormap

    bars = plt.bar(x, ateco_counts.values, color=colors, edgecolor='white', linewidth=0.5)

    # Customize the plot
    plt.xlabel('2-Digit ATECO Codes', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Companies with No Adoption', fontsize=12, fontweight='bold')
    plt.title('Distribution of Companies with No Framework Adoption\nby 2-Digit ATECO Code (2022-2024)',
              fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    plt.xticks(x, ateco_counts.index, fontsize=10, rotation=0)
    plt.yticks(fontsize=10)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, color='black')

    # Add a subtle grid and fancy border
    plt.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()
    # Show and save the plot
    ##plt.show()
    plt.savefig('fig1.png')

def figure2(df):
    # Load and process the dataset
    data = df.copy()

    # Create adoption categories for 2024
    data['ESRS_only'] = ((data['ESRS_2024'] == 1) & (data['SASB_2024'] == 0)).astype(int)
    data['SASB_only'] = ((data['SASB_2024'] == 1) & (data['ESRS_2024'] == 0)).astype(int)
    data['Both'] = ((data['ESRS_2024'] == 1) & (data['SASB_2024'] == 1)).astype(int)

    # Filter companies that have at least one of the frameworks
    esrs_sasb_adopters = data[(data['ESRS_2024'] == 1) | (data['SASB_2024'] == 1)].copy()

    # Extract 2-digit ATECO codes
    esrs_sasb_adopters['Ateco_2digit'] = esrs_sasb_adopters['Ateco'].astype(str).str[:2].fillna('na')

    # Group by ATECO code and count adoption types
    ateco_adoption = esrs_sasb_adopters.groupby('Ateco_2digit').agg({
        'ESRS_only': 'sum',
        'SASB_only': 'sum',
        'Both': 'sum'
    }).fillna(0)

    # Calculate total adopters per ATECO code
    ateco_adoption['Total'] = ateco_adoption.sum(axis=1)
    ateco_adoption = ateco_adoption.sort_values('Total', ascending=False)

    # Get ATECO descriptions for legend
    ateco_descriptions = data[['Ateco', 'AtecoX']].drop_duplicates().set_index('Ateco')['AtecoX'].to_dict()
    ateco_labels = {code: ateco_descriptions.get(code, 'Not Available') for code in ateco_adoption.index}

    # Create figure
    plt.figure(figsize=(14, 8), facecolor='#f0f8ff')
    ax = plt.gca()
    ax.set_facecolor('#e6f0fa')

    # Create stacked bar plot
    x = np.arange(len(ateco_adoption))
    bottom = np.zeros(len(ateco_adoption))

    # Define colors for each category
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # ESRS_only, SASB_only, Both

    bars1 = plt.bar(x, ateco_adoption['ESRS_only'], color=colors[0],
                    edgecolor='white', linewidth=0.5, label='ESRS Only')
    bottom += ateco_adoption['ESRS_only']

    bars2 = plt.bar(x, ateco_adoption['SASB_only'], bottom=bottom, color=colors[1],
                    edgecolor='white', linewidth=0.5, label='SASB Only')
    bottom += ateco_adoption['SASB_only']

    bars3 = plt.bar(x, ateco_adoption['Both'], bottom=bottom, color=colors[2],
                    edgecolor='white', linewidth=0.5, label='Both Frameworks')

    # Customize the plot
    plt.xlabel('2-Digit ATECO Codes', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Companies', fontsize=12, fontweight='bold')
    plt.title('Distribution of ESRS 2024 and SASB 2024 Framework Adoption\nby 2-Digit ATECO Code',
              fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    plt.xticks(x, ateco_adoption.index, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)

    # Add value labels on bars (total per ATECO code)
    for i, (idx, row) in enumerate(ateco_adoption.iterrows()):
        total = row['Total']
        if total > 0:
            ax.text(i, total + 0.1, f'{int(total)}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#2c3e50')

    # Add legend
    plt.legend(title='Adoption Type', title_fontsize=10, fontsize=9,
               loc='upper right', framealpha=0.9)

    # Add a subtle grid and fancy border
    plt.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5, axis='y')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Print some statistics
    total_esrs_only = esrs_sasb_adopters['ESRS_only'].sum()
    total_sasb_only = esrs_sasb_adopters['SASB_only'].sum()
    total_both = esrs_sasb_adopters['Both'].sum()
    total_adopters = len(esrs_sasb_adopters)

    print(f"Adoption Statistics (2024):")
    print(f"Total companies with ESRS or SASB: {total_adopters}")
    print(f"ESRS only: {total_esrs_only} ({total_esrs_only / total_adopters * 100:.1f}%)")
    print(f"SASB only: {total_sasb_only} ({total_sasb_only / total_adopters * 100:.1f}%)")
    print(f"Both frameworks: {total_both} ({total_both / total_adopters * 100:.1f}%)")

    # Show and save the plot
    plt.savefig('fig2.png', dpi=300, bbox_inches='tight')
    #plt.show()

def figure3(df):
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Display basic info
    print("Dataset Overview:")
    print(f"Total companies: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())

    # Identify reporting standard columns
    reporting_cols = ['GRI_2022', 'ESRS_2022', 'SASB_2022',
                      'GRI_2023', 'ESRS_2023', 'SASB_2023',
                      'GRI_2024', 'ESRS_2024', 'SASB_2024']

    # Create a flag for companies with all 9 values = 0
    df['all_zeros'] = (df[reporting_cols].sum(axis=1) == 0)

    print(f"\nCompanies with all zeros: {df['all_zeros'].sum()} ({df['all_zeros'].mean() * 100:.1f}%)")

    # Analyze by Ateco sector (2-digit code)
    # First, let's clean the Ateco column and convert to string
    df['Ateco'] = df['Ateco'].astype(str)

    # Get the sector description for each Ateco code
    sector_mapping = df[['Ateco', 'AtecoX']].drop_duplicates().set_index('Ateco')['AtecoX'].to_dict()

    # Calculate statistics by sector
    sector_stats = df.groupby('Ateco').agg({
        'all_zeros': ['count', 'sum', 'mean'],
        'Name': 'first'
    }).round(3)

    # Flatten column names
    sector_stats.columns = ['total_companies', 'zero_companies', 'zero_percentage', 'dummy']
    sector_stats = sector_stats.drop('dummy', axis=1)

    # Calculate weighted percentage (percentage of total zeros coming from each sector)
    total_zeros = sector_stats['zero_companies'].sum()
    sector_stats['weighted_percentage'] = (sector_stats['zero_companies'] / total_zeros * 100).round(1)

    # Add sector descriptions
    sector_stats['sector_description'] = sector_stats.index.map(sector_mapping)

    # Filter out sectors with no data or very few companies
    sector_stats = sector_stats[sector_stats['total_companies'] >= 1]

    print(f"\nSectors with highest absolute number of zero-reporting companies:")
    print(sector_stats.nlargest(10, 'zero_companies')[
              ['total_companies', 'zero_companies', 'zero_percentage', 'weighted_percentage', 'sector_description']])

    print(f"\nSectors with highest percentage of zero-reporting companies:")
    high_percentage_sectors = sector_stats[sector_stats['total_companies'] >= 3].nlargest(10, 'zero_percentage')
    print(high_percentage_sectors[
              ['total_companies', 'zero_companies', 'zero_percentage', 'weighted_percentage', 'sector_description']])

    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))

    # 1. Bar plot: Weighted percentage of zero-reporting companies by sector
    top_sectors_weighted = sector_stats.nlargest(15, 'weighted_percentage')
    bars1 = ax1.bar(range(len(top_sectors_weighted)),
                    top_sectors_weighted['weighted_percentage'],
                    color='lightcoral', edgecolor='darkred', linewidth=0.5)

    ax1.set_title('A. Sectors Contributing Most to Zero-Reporting Companies\n(Weighted by Percentage of Total Zeros)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Percentage of Total Zero-Reporting Companies (%)', fontsize=12)
    ax1.set_xlabel('ATECO Sector Code', fontsize=12)

    # Add sector labels with descriptions (truncated for readability)
    labels = [f"{idx}\n{desc[:25]}..." if len(desc) > 25 else f"{idx}\n{desc}"
              for idx, desc in zip(top_sectors_weighted.index, top_sectors_weighted['sector_description'])]
    ax1.set_xticks(range(len(top_sectors_weighted)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    # Add value labels on bars
    for bar, value in zip(bars1, top_sectors_weighted['weighted_percentage']):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Bar plot: Percentage of companies with zeros within each sector
    top_sectors_rate = sector_stats[sector_stats['total_companies'] >= 3].nlargest(15, 'zero_percentage')
    bars2 = ax2.bar(range(len(top_sectors_rate)),
                    top_sectors_rate['zero_percentage'] * 100,
                    color='skyblue', edgecolor='navy', linewidth=0.5)

    ax2.set_title('B. Sectors with Highest Rate of Zero-Reporting Companies\n(Percentage within Sector)',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Percentage of Companies with Zero Reporting (%)', fontsize=12)
    ax2.set_xlabel('ATECO Sector Code', fontsize=12)
    ax2.set_ylim(0, 110)

    # Add sector labels
    labels2 = [f"{idx}\n{desc[:25]}..." if len(desc) > 25 else f"{idx}\n{desc}"
               for idx, desc in zip(top_sectors_rate.index, top_sectors_rate['sector_description'])]
    ax2.set_xticks(range(len(top_sectors_rate)))
    ax2.set_xticklabels(labels2, rotation=45, ha='right', fontsize=9)

    # Add value labels and sample size
    for i, (bar, value, total) in enumerate(
            zip(bars2, top_sectors_rate['zero_percentage'] * 100, top_sectors_rate['total_companies'])):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{value:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(bar.get_x() + bar.get_width() / 2, -5,
                 f'n={int(total)}', ha='center', va='top', fontsize=8, color='gray')

    # 3. Pie chart: Distribution of zero-reporting companies across sectors
    # Group smaller sectors into "Others"
    pie_data = sector_stats.nlargest(8, 'weighted_percentage').copy()
    others_sum = sector_stats['weighted_percentage'].sum() - pie_data['weighted_percentage'].sum()
    others_row = pd.DataFrame({
        'weighted_percentage': [others_sum],
        'sector_description': ['Other Sectors'],
        'zero_companies': [sector_stats['zero_companies'].sum() - pie_data['zero_companies'].sum()]
    }, index=['Other'])

    pie_data = pd.concat([pie_data, others_row])

    # Create pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
    wedges, texts, autotexts = ax3.pie(pie_data['weighted_percentage'],
                                       labels=pie_data.index,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=colors,
                                       textprops={'fontsize': 10})

    ax3.set_title('C. Distribution of Zero-Reporting Companies\nAcross Sectors (Weighted)',
                  fontsize=14, fontweight='bold', pad=20)

    # Improve pie chart labels
    for i, (wedge, text) in enumerate(zip(wedges, texts)):
        sector_idx = pie_data.index[i]
        desc = pie_data.loc[sector_idx, 'sector_description'][:20] + '...' if len(
            pie_data.loc[sector_idx, 'sector_description']) > 20 else pie_data.loc[sector_idx, 'sector_description']
        text.set_text(f"{sector_idx}\n{desc}")

    # 4. Overall statistics and summary
    ax4.axis('off')
    summary_text = (
        f"ANALYSIS SUMMARY\n\n"
        f"Total Companies: {len(df):,}\n"
        f"Zero-Reporting Companies: {df['all_zeros'].sum():,}\n"
        f"Overall Rate: {df['all_zeros'].mean() * 100:.1f}%\n\n"
        f"KEY FINDINGS:\n"
        f"• {sector_stats.loc['70', 'zero_companies']:.0f} zero-reporting companies\n  in Sector 70 (Management Consulting)\n"
        f"• {sector_stats.loc['64', 'zero_companies']:.0f} zero-reporting companies\n  in Sector 64 (Financial Services)\n"
        f"• {sector_stats.loc['46', 'zero_companies']:.0f} zero-reporting companies\n  in Sector 46 (Wholesale Trade)\n\n"
        f"These 3 sectors account for\n{sector_stats.loc[['70', '64', '46'], 'weighted_percentage'].sum():.1f}% of all\nzero-reporting companies"
    )

    ax4.text(0.1, 0.9, summary_text, fontsize=12, va='top', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))

    plt.tight_layout()
    plt.savefig('fig3.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Additional detailed table for the article
    print("\n" + "=" * 80)
    print("DETAILED SECTOR ANALYSIS FOR ACADEMIC ARTICLE")
    print("=" * 80)

    detailed_table = sector_stats.nlargest(15, 'weighted_percentage')[
        ['total_companies', 'zero_companies', 'zero_percentage', 'weighted_percentage', 'sector_description']].copy()
    detailed_table['zero_percentage'] = (detailed_table['zero_percentage'] * 100).round(1)
    detailed_table = detailed_table.rename(columns={
        'total_companies': 'Total Companies',
        'zero_companies': 'Zero-Reporting',
        'zero_percentage': 'Sector Rate (%)',
        'weighted_percentage': 'Weighted Share (%)',
        'sector_description': 'Sector Description'
    })

    print(detailed_table.to_string(max_colwidth=40))

    # Calculate some additional statistics for the article
    print(f"\nADDITIONAL INSIGHTS:")
    print(f"- Sectors with 100% zero-reporting rate: {len(sector_stats[sector_stats['zero_percentage'] == 1])}")
    print(
        f"- Sectors with mixed reporting: {len(sector_stats[(sector_stats['zero_percentage'] > 0) & (sector_stats['zero_percentage'] < 1)])}")
    print(f"- Sectors with no zero-reporting: {len(sector_stats[sector_stats['zero_percentage'] == 0])}")

    # Show sectors that are completely non-compliant (100% zeros)
    completely_non_compliant = sector_stats[
        (sector_stats['zero_percentage'] == 1) & (sector_stats['total_companies'] >= 2)]
    if not completely_non_compliant.empty:
        print(f"\nSECTORS WITH 100% ZERO-REPORTING (≥2 companies):")
        for idx, row in completely_non_compliant.iterrows():
            print(f"  - Sector {idx}: {row['sector_description']} ({int(row['total_companies'])} companies)")

def figure4(df):
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Identify reporting standard columns
    reporting_cols = ['GRI_2022', 'ESRS_2022', 'SASB_2022',
                      'GRI_2023', 'ESRS_2023', 'SASB_2023',
                      'GRI_2024', 'ESRS_2024', 'SASB_2024']

    # Create a flag for companies with all 9 values = 0
    df['all_zeros'] = (df[reporting_cols].sum(axis=1) == 0)

    # Analyze by Ateco sector (2-digit code)
    df['Ateco'] = df['Ateco'].astype(str)

    # Calculate statistics by sector
    sector_stats = df.groupby('Ateco').agg({
        'all_zeros': ['count', 'sum', 'mean']
    }).round(3)

    # Flatten column names
    sector_stats.columns = ['total_companies', 'zero_companies', 'zero_percentage']
    sector_stats = sector_stats[sector_stats['total_companies'] >= 1]

    # Calculate weighted percentage
    total_zeros = sector_stats['zero_companies'].sum()
    sector_stats['weighted_percentage'] = (sector_stats['zero_companies'] / total_zeros * 100).round(1)

    # GRAPH 1: Weighted percentage of zero-reporting companies by sector
    plt.figure(figsize=(12, 8))
    top_sectors_weighted = sector_stats.nlargest(15, 'weighted_percentage')
    bars1 = plt.bar(range(len(top_sectors_weighted)),
                    top_sectors_weighted['weighted_percentage'],
                    color='lightcoral', edgecolor='darkred', linewidth=0.5)

    plt.title('Sectors Contributing Most to Zero-Reporting Companies\n(Weighted by Percentage of Total Zeros)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Percentage of Total Zero-Reporting Companies (%)', fontsize=12)
    plt.xlabel('ATECO Sector Code', fontsize=12)

    # Use only Ateco codes as labels
    labels = [f"{idx}" for idx in top_sectors_weighted.index]
    plt.xticks(range(len(top_sectors_weighted)), labels, rotation=45, ha='right', fontsize=11)

    # Add value labels on bars
    for bar, value in zip(bars1, top_sectors_weighted['weighted_percentage']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('weighted_percentage_by_sector.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # GRAPH 2: Percentage of companies with zeros within each sector
    plt.figure(figsize=(12, 8))
    top_sectors_rate = sector_stats[sector_stats['total_companies'] >= 3].nlargest(15, 'zero_percentage')
    bars2 = plt.bar(range(len(top_sectors_rate)),
                    top_sectors_rate['zero_percentage'] * 100,
                    color='skyblue', edgecolor='navy', linewidth=0.5)

    plt.title('Sectors with Highest Rate of Zero-Reporting Companies\n(Percentage within Sector)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Percentage of Companies with Zero Reporting (%)', fontsize=12)
    plt.xlabel('ATECO Sector Code', fontsize=12)
    plt.ylim(0, 110)

    # Use only Ateco codes as labels
    labels2 = [f"{idx}" for idx in top_sectors_rate.index]
    plt.xticks(range(len(top_sectors_rate)), labels2, rotation=45, ha='right', fontsize=11)

    # Add value labels and sample size
    for i, (bar, value, total) in enumerate(
            zip(bars2, top_sectors_rate['zero_percentage'] * 100, top_sectors_rate['total_companies'])):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{value:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(bar.get_x() + bar.get_width() / 2, -5,
                 f'n={int(total)}', ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig('sector_zero_rates.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # GRAPH 3: Pie chart - Distribution of zero-reporting companies across sectors
    plt.figure(figsize=(10, 8))

    # Group smaller sectors into "Others"
    pie_data = sector_stats.nlargest(8, 'weighted_percentage').copy()
    others_sum = sector_stats['weighted_percentage'].sum() - pie_data['weighted_percentage'].sum()
    others_row = pd.DataFrame({
        'weighted_percentage': [others_sum],
        'zero_companies': [sector_stats['zero_companies'].sum() - pie_data['zero_companies'].sum()]
    }, index=['Other'])

    pie_data = pd.concat([pie_data, others_row])

    # Create pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
    wedges, texts, autotexts = plt.pie(pie_data['weighted_percentage'],
                                       labels=pie_data.index,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=colors,
                                       textprops={'fontsize': 11})

    plt.title('Distribution of Zero-Reporting Companies\nAcross Sectors (Weighted)',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('sector_distribution_pie.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Print summary statistics
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total Companies: {len(df):,}")
    print(f"Zero-Reporting Companies: {df['all_zeros'].sum():,}")
    print(f"Overall Zero-Reporting Rate: {df['all_zeros'].mean() * 100:.1f}%")
    print(f"Total Sectors (Ateco codes): {len(sector_stats)}")

    print(f"\nTop Sectors by Weighted Contribution:")
    top_weighted = sector_stats.nlargest(5, 'weighted_percentage')
    for idx, row in top_weighted.iterrows():
        print(f"  Sector {idx}: {row['weighted_percentage']:.1f}% of total zeros")

    print(f"\nTop Sectors by Zero-Reporting Rate (min. 3 companies):")
    top_rates = sector_stats[sector_stats['total_companies'] >= 3].nlargest(5, 'zero_percentage')
    for idx, row in top_rates.iterrows():
        print(f"  Sector {idx}: {row['zero_percentage'] * 100:.1f}% zeros (n={int(row['total_companies'])})")

def figure5(df):
    # Set professional style with thinner grids
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Customize grid to be thinner and less aggressive
    plt.rcParams['grid.linewidth'] = 0.3
    plt.rcParams['grid.alpha'] = 0.5

    # Identify reporting standard columns
    reporting_cols = ['GRI_2022', 'ESRS_2022', 'SASB_2022',
                      'GRI_2023', 'ESRS_2023', 'SASB_2023',
                      'GRI_2024', 'ESRS_2024', 'SASB_2024']

    # Create a flag for companies with all 9 values = 0
    df['all_zeros'] = (df[reporting_cols].sum(axis=1) == 0)

    # Analyze by Ateco sector (2-digit code)
    df['Ateco'] = df['Ateco'].astype(str)

    # Calculate statistics by sector
    sector_stats = df.groupby('Ateco').agg({
        'all_zeros': ['count', 'sum', 'mean']
    }).round(3)

    # Flatten column names
    sector_stats.columns = ['total_companies', 'zero_companies', 'zero_percentage']
    sector_stats = sector_stats[sector_stats['total_companies'] >= 1]

    # Calculate weighted percentage
    total_zeros = sector_stats['zero_companies'].sum()
    sector_stats['weighted_percentage'] = (sector_stats['zero_companies'] / total_zeros * 100).round(1)

    # ANALYSIS 1: Weighted percentage of zero-reporting companies by sector

    # Bar plot for Analysis 1
    plt.figure(figsize=(12, 8))
    top_sectors_weighted = sector_stats.nlargest(15, 'weighted_percentage')
    bars1 = plt.bar(range(len(top_sectors_weighted)),
                    top_sectors_weighted['weighted_percentage'],
                    color='lightcoral', edgecolor='darkred', linewidth=0.5)

    plt.title('Sectors Contributing Most to Zero-Reporting Companies\n(Weighted by Percentage of Total Zeros)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Percentage of Total Zero-Reporting Companies (%)', fontsize=12)
    plt.xlabel('ATECO Sector Code', fontsize=12)

    # Use only Ateco codes as labels
    labels = [f"{idx}" for idx in top_sectors_weighted.index]
    plt.xticks(range(len(top_sectors_weighted)), labels, rotation=45, ha='right', fontsize=11)

    # Add value labels on bars
    for bar, value in zip(bars1, top_sectors_weighted['weighted_percentage']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('weighted_percentage_by_sector.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Pie chart for Analysis 1
    plt.figure(figsize=(10, 8))

    # Group smaller sectors into "Others" for pie chart
    pie_data1 = sector_stats.nlargest(8, 'weighted_percentage').copy()
    others_sum1 = sector_stats['weighted_percentage'].sum() - pie_data1['weighted_percentage'].sum()
    others_row1 = pd.DataFrame({
        'weighted_percentage': [others_sum1],
        'zero_companies': [sector_stats['zero_companies'].sum() - pie_data1['zero_companies'].sum()]
    }, index=['Other'])

    pie_data1 = pd.concat([pie_data1, others_row1])

    # Create pie chart
    colors1 = plt.cm.Set3(np.linspace(0, 1, len(pie_data1)))
    wedges1, texts1, autotexts1 = plt.pie(pie_data1['weighted_percentage'],
                                          labels=pie_data1.index,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          colors=colors1,
                                          textprops={'fontsize': 11})

    plt.title('Distribution of Zero-Reporting Companies\nAcross Sectors (Weighted)',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('sector_distribution_weighted_pie.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # ANALYSIS 2: Percentage of companies with zeros within each sector

    # Bar plot for Analysis 2
    plt.figure(figsize=(12, 8))
    top_sectors_rate = sector_stats[sector_stats['total_companies'] >= 3].nlargest(15, 'zero_percentage')
    bars2 = plt.bar(range(len(top_sectors_rate)),
                    top_sectors_rate['zero_percentage'] * 100,
                    color='skyblue', edgecolor='navy', linewidth=0.5)

    plt.title('Sectors with Highest Rate of Zero-Reporting Companies\n(Percentage within Sector)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Percentage of Companies with Zero Reporting (%)', fontsize=12)
    plt.xlabel('ATECO Sector Code', fontsize=12)
    plt.ylim(0, 110)

    # Use only Ateco codes as labels - increased bottom margin to prevent overlap
    labels2 = [f"{idx}" for idx in top_sectors_rate.index]
    plt.xticks(range(len(top_sectors_rate)), labels2, rotation=45, ha='right', fontsize=11)

    # Add value labels and sample size - adjusted positions to prevent overlap
    for i, (bar, value, total) in enumerate(
            zip(bars2, top_sectors_rate['zero_percentage'] * 100, top_sectors_rate['total_companies'])):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{value:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Move n= labels further down and make them smaller to prevent overlap with x-axis labels
        plt.text(bar.get_x() + bar.get_width() / 2, -8,
                 f'n={int(total)}', ha='center', va='top', fontsize=8, color='gray', alpha=0.8)

    # Adjust subplot parameters to give more room at the bottom
    plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()
    plt.savefig('sector_zero_rates.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Pie chart for Analysis 2 - Showing more sectors without "Others" category
    plt.figure(figsize=(12, 10))

    # Get top sectors by zero rate (with at least 3 companies) - show more sectors
    top_pie_sectors = sector_stats[sector_stats['total_companies'] >= 3].nlargest(12, 'zero_percentage')

    # Create custom labels showing sector code, rate, and sample size
    def make_rate_label(idx, row):
        return f"Sector {idx}\n{row['zero_percentage'] * 100:.0f}% (n={int(row['total_companies'])})"

    labels_pie2 = [make_rate_label(idx, row) for idx, row in top_pie_sectors.iterrows()]

    # Use the zero percentage as the size for the pie chart
    sizes = top_pie_sectors['zero_percentage'] * 100

    # Create a more distinct color palette
    colors2 = plt.cm.tab20(np.linspace(0, 1, len(top_pie_sectors)))

    # Print summary statistics
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total Companies: {len(df):,}")
    print(f"Zero-Reporting Companies: {df['all_zeros'].sum():,}")
    print(f"Overall Zero-Reporting Rate: {df['all_zeros'].mean() * 100:.1f}%")
    print(f"Total Sectors (Ateco codes): {len(sector_stats)}")

    print(f"\nTop Sectors by Weighted Contribution:")
    top_weighted = sector_stats.nlargest(5, 'weighted_percentage')
    for idx, row in top_weighted.iterrows():
        print(f"  Sector {idx}: {row['weighted_percentage']:.1f}% of total zeros")

    print(f"\nTop Sectors by Zero-Reporting Rate (min. 3 companies):")
    top_rates = sector_stats[sector_stats['total_companies'] >= 3].nlargest(5, 'zero_percentage')
    for idx, row in top_rates.iterrows():
        print(f"  Sector {idx}: {row['zero_percentage'] * 100:.1f}% zeros (n={int(row['total_companies'])})")

    # Additional statistics about high zero rate sectors
    print(f"\nSectors with 100% Zero-Reporting (any sample size):")
    hundred_percent_sectors = sector_stats[sector_stats['zero_percentage'] == 1]
    for idx, row in hundred_percent_sectors.iterrows():
        print(f"  Sector {idx}: {int(row['total_companies'])} companies")

def step1(df_Extra, df):
    df_Extra = pd.read_csv('../Part 2 (AA1000 only)/ATECO_codes.csv')
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
        #print(rows)

    # Write the modified data
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"File processed successfully. Output saved to: {output_file}")

def figure1_1(df):
    # Load and process the dataset
    data = df
    framework_columns = ['GRI_2022', 'ESRS_2022', 'SASB_2022', 'GRI_2023', 'ESRS_2023', 'SASB_2023', 'GRI_2024',
                         'ESRS_2024', 'SASB_2024']
    data['Total_Adoption'] = data[framework_columns].sum(axis=1)
    zero_adoption = data[data['Total_Adoption'] == 0].copy()

    # Extract Ateco section Codes and count occurrences
    zero_adoption['ateco_letter'] = zero_adoption['ateco']
    ateco_counts = zero_adoption['ateco_letter'].value_counts().sort_index()

    # ATECOX descriptions for legend (unique 2-digit codes)
    ateco_descriptions = data[['ateco', 'atecoX']].drop_duplicates().set_index('ateco')['atecoX'].to_dict()
    ateco_labels = {code: ateco_descriptions.get(code, 'Not Available') for code in ateco_counts.index}

    # Create figure with a cool background
    plt.figure(figsize=(12, 6), facecolor='#f0f8ff')
    ax = plt.gca()
    ax.set_facecolor('#e6f0fa')

    # Bar plot with reversed gradient colors (red for high, green for low)
    x = np.arange(len(ateco_counts))
    # Normalize counts for color scaling (reversed: high = red, low = green)
    norm_counts = ateco_counts.values / ateco_counts.max()
    colors = plt.cm.RdYlGn_r(norm_counts)  # Reversed RdYlGn colormap

    bars = plt.bar(x, ateco_counts.values, color=colors, edgecolor='white', linewidth=0.5)

    # Customize the plot
    plt.xlabel('Ateco Section', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Companies with No Adoption', fontsize=12, fontweight='bold')
    plt.title('Distribution of Companies with No Framework Adoption\nby Ateco section (2022-2024)',
              fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    plt.xticks(x, ateco_counts.index, fontsize=10, rotation=0)
    plt.yticks(fontsize=10)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, color='black')

    # Add a subtle grid and fancy border
    plt.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()
    # Show and save the plot
    ##plt.show()
    plt.savefig('fig1_1.png')

def figure2_1(df):
    # Load and process the dataset
    data = df.copy()

    # Create adoption categories for 2024
    data['ESRS_only'] = ((data['ESRS_2024'] == 1) & (data['SASB_2024'] == 0)).astype(int)
    data['SASB_only'] = ((data['SASB_2024'] == 1) & (data['ESRS_2024'] == 0)).astype(int)
    data['Both'] = ((data['ESRS_2024'] == 1) & (data['SASB_2024'] == 1)).astype(int)

    # Filter companies that have at least one of the frameworks
    esrs_sasb_adopters = data[(data['ESRS_2024'] == 1) | (data['SASB_2024'] == 1)].copy()

    # Extract Ateco section Codes
    esrs_sasb_adopters['ateco'] = esrs_sasb_adopters['ateco']

    # Group by ATECO code and count adoption types
    ateco_adoption = esrs_sasb_adopters.groupby('ateco').agg({
        'ESRS_only': 'sum',
        'SASB_only': 'sum',
        'Both': 'sum'
    }).fillna(0)

    # Calculate total adopters per ATECO code
    ateco_adoption['Total'] = ateco_adoption.sum(axis=1)
    ateco_adoption = ateco_adoption.sort_values('Total', ascending=False)

    # Get ATECO descriptions for legend
    ateco_descriptions = data[['ateco', 'atecoX']].drop_duplicates().set_index('ateco')['atecoX'].to_dict()
    ateco_labels = {code: ateco_descriptions.get(code, 'Not Available') for code in ateco_adoption.index}

    # Create figure
    plt.figure(figsize=(14, 8), facecolor='#f0f8ff')
    ax = plt.gca()
    ax.set_facecolor('#e6f0fa')

    # Create stacked bar plot
    x = np.arange(len(ateco_adoption))
    bottom = np.zeros(len(ateco_adoption))

    # Define colors for each category
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # ESRS_only, SASB_only, Both

    bars1 = plt.bar(x, ateco_adoption['ESRS_only'], color=colors[0],
                    edgecolor='white', linewidth=0.5, label='ESRS Only')
    bottom += ateco_adoption['ESRS_only']

    bars2 = plt.bar(x, ateco_adoption['SASB_only'], bottom=bottom, color=colors[1],
                    edgecolor='white', linewidth=0.5, label='SASB Only')
    bottom += ateco_adoption['SASB_only']

    bars3 = plt.bar(x, ateco_adoption['Both'], bottom=bottom, color=colors[2],
                    edgecolor='white', linewidth=0.5, label='Both Frameworks')

    # Customize the plot
    plt.xlabel('Ateco section Codes', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Companies', fontsize=12, fontweight='bold')
    plt.title('Distribution of ESRS 2024 and SASB 2024 Framework Adoption\nby Ateco section Code',
              fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
    plt.xticks(x, ateco_adoption.index, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)

    # Add value labels on bars (total per ATECO code)
    for i, (idx, row) in enumerate(ateco_adoption.iterrows()):
        total = row['Total']
        if total > 0:
            ax.text(i, total + 0.1, f'{int(total)}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#2c3e50')

    # Add legend
    plt.legend(title='Adoption Type', title_fontsize=10, fontsize=9,
               loc='upper right', framealpha=0.9)

    # Add a subtle grid and fancy border
    plt.grid(True, which='both', linestyle='--', alpha=0.3, linewidth=0.5, axis='y')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Print some statistics
    total_esrs_only = esrs_sasb_adopters['ESRS_only'].sum()
    total_sasb_only = esrs_sasb_adopters['SASB_only'].sum()
    total_both = esrs_sasb_adopters['Both'].sum()
    total_adopters = len(esrs_sasb_adopters)

    print(f"Adoption Statistics (2024):")
    print(f"Total companies with ESRS or SASB: {total_adopters}")
    print(f"ESRS only: {total_esrs_only} ({total_esrs_only / total_adopters * 100:.1f}%)")
    print(f"SASB only: {total_sasb_only} ({total_sasb_only / total_adopters * 100:.1f}%)")
    print(f"Both frameworks: {total_both} ({total_both / total_adopters * 100:.1f}%)")

    # Show and save the plot
    plt.savefig('fig2_1.png', dpi=300, bbox_inches='tight')
    #plt.show()

def figure4_1(df):
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Identify reporting standard columns
    reporting_cols = ['GRI_2022', 'ESRS_2022', 'SASB_2022',
                      'GRI_2023', 'ESRS_2023', 'SASB_2023',
                      'GRI_2024', 'ESRS_2024', 'SASB_2024']

    # Create a flag for companies with all 9 values = 0
    df['all_zeros'] = (df[reporting_cols].sum(axis=1) == 0)

    # Analyze by Ateco sector (2-digit code)
    df['ateco'] = df['ateco'].astype(str)

    # Calculate statistics by sector
    sector_stats = df.groupby('ateco').agg({
        'all_zeros': ['count', 'sum', 'mean']
    }).round(3)

    # Flatten column names
    sector_stats.columns = ['total_companies', 'zero_companies', 'zero_percentage']
    sector_stats = sector_stats[sector_stats['total_companies'] >= 1]

    # Calculate weighted percentage
    total_zeros = sector_stats['zero_companies'].sum()
    sector_stats['weighted_percentage'] = (sector_stats['zero_companies'] / total_zeros * 100).round(1)

    # GRAPH 1: Weighted percentage of zero-reporting companies by sector
    plt.figure(figsize=(12, 8))
    top_sectors_weighted = sector_stats.nlargest(15, 'weighted_percentage')
    bars1 = plt.bar(range(len(top_sectors_weighted)),
                    top_sectors_weighted['weighted_percentage'],
                    color='lightcoral', edgecolor='darkred', linewidth=0.5)

    plt.title('Sections Contributing Most to Zero-Reporting Companies\n(Weighted by Percentage of Total Zeros)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Percentage of Total Zero-Reporting Companies (%)', fontsize=12)
    plt.xlabel('Ateco Section Code', fontsize=12)

    # Use only Ateco codes as labels
    labels = [f"{idx}" for idx in top_sectors_weighted.index]
    plt.xticks(range(len(top_sectors_weighted)), labels, rotation=45, ha='right', fontsize=11)

    # Add value labels on bars
    for bar, value in zip(bars1, top_sectors_weighted['weighted_percentage']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('weighted_percentage_by_section.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # GRAPH 2: Percentage of companies with zeros within each sector
    plt.figure(figsize=(12, 8))
    top_sectors_rate = sector_stats[sector_stats['total_companies'] >= 3].nlargest(15, 'zero_percentage')
    bars2 = plt.bar(range(len(top_sectors_rate)),
                    top_sectors_rate['zero_percentage'] * 100,
                    color='skyblue', edgecolor='navy', linewidth=0.5)

    plt.title('Sections with Highest Rate of Zero-Reporting Companies\n(Percentage within Section)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Percentage of Companies with Zero Reporting (%)', fontsize=12)
    plt.xlabel('Ateco section Code', fontsize=12)
    plt.ylim(0, 110)

    # Use only Ateco codes as labels
    labels2 = [f"{idx}" for idx in top_sectors_rate.index]
    plt.xticks(range(len(top_sectors_rate)), labels2, rotation=45, ha='right', fontsize=11)

    # Add value labels and sample size
    for i, (bar, value, total) in enumerate(
            zip(bars2, top_sectors_rate['zero_percentage'] * 100, top_sectors_rate['total_companies'])):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{value:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(bar.get_x() + bar.get_width() / 2, -5,
                 f'n={int(total)}', ha='center', va='top', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig('section_zero_rates.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # GRAPH 3: Pie chart - Distribution of zero-reporting companies across sectors
    plt.figure(figsize=(10, 8))

    # Group smaller sectors into "Others"
    pie_data = sector_stats.nlargest(8, 'weighted_percentage').copy()
    others_sum = sector_stats['weighted_percentage'].sum() - pie_data['weighted_percentage'].sum()
    others_row = pd.DataFrame({
        'weighted_percentage': [others_sum],
        'zero_companies': [sector_stats['zero_companies'].sum() - pie_data['zero_companies'].sum()]
    }, index=['Other'])

    pie_data = pd.concat([pie_data, others_row])

    # Create pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
    wedges, texts, autotexts = plt.pie(pie_data['weighted_percentage'],
                                       labels=pie_data.index,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=colors,
                                       textprops={'fontsize': 11})

    plt.title('Distribution of Zero-Reporting Companies\nAcross Sections (Weighted)',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('section_distribution_pie.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Print summary statistics
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total Companies: {len(df):,}")
    print(f"Zero-Reporting Companies: {df['all_zeros'].sum():,}")
    print(f"Overall Zero-Reporting Rate: {df['all_zeros'].mean() * 100:.1f}%")
    print(f"Total Sections (Ateco codes): {len(sector_stats)}")

    print(f"\nTop Sections by Weighted Contribution:")
    top_weighted = sector_stats.nlargest(5, 'weighted_percentage')
    for idx, row in top_weighted.iterrows():
        print(f"  Section {idx}: {row['weighted_percentage']:.1f}% of total zeros")

    print(f"\nTop Sections by Zero-Reporting Rate (min. 3 companies):")
    top_rates = sector_stats[sector_stats['total_companies'] >= 3].nlargest(5, 'zero_percentage')
    for idx, row in top_rates.iterrows():
        print(f"  Section {idx}: {row['zero_percentage'] * 100:.1f}% zeros (n={int(row['total_companies'])})")

def step2():
    step2_df = pd.read_csv('step2.csv')
    tidier_df = pd.read_csv('Tidier_Dataset.csv')

    # Create a mapping dictionary for ATECO codes to letters and descriptions
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
    tidier_df[['ateco_letter', 'ateco_description']] = tidier_df['Ateco'].apply(
        lambda x: pd.Series(get_ateco_info(x))
    )

    # Reorder columns to place the new columns after 'Name'
    cols = list(tidier_df.columns)
    name_idx = cols.index('Name')

    # Remove the new columns from their current position
    cols.remove('ateco_letter')
    cols.remove('ateco_description')

    # Insert them after 'Name'
    cols.insert(name_idx + 1, 'ateco_letter')
    cols.insert(name_idx + 2, 'ateco_description')

    tidier_df = tidier_df[cols]

    # Rename the columns as requested
    tidier_df = tidier_df.rename(columns={
        'ateco_letter': 'ateco',
        'ateco_description': 'atecoX'
    })

    # Save the updated dataset
    tidier_df.to_csv('Starting_Dataset.csv.csv', index=False)

def figure5_1(df):
    # Set professional style with thinner grids
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Customize grid to be thinner and less aggressive
    plt.rcParams['grid.linewidth'] = 0.3
    plt.rcParams['grid.alpha'] = 0.5

    # Identify reporting standard columns
    reporting_cols = ['GRI_2022', 'ESRS_2022', 'SASB_2022',
                      'GRI_2023', 'ESRS_2023', 'SASB_2023',
                      'GRI_2024', 'ESRS_2024', 'SASB_2024']

    # Create a flag for companies with all 9 values = 0
    df['all_zeros'] = (df[reporting_cols].sum(axis=1) == 0)

    # Analyze by Ateco sector (2-digit code)
    df['ateco'] = df['ateco'].astype(str)

    # Calculate statistics by sector
    sector_stats = df.groupby('ateco').agg({
        'all_zeros': ['count', 'sum', 'mean']
    }).round(3)

    # Flatten column names
    sector_stats.columns = ['total_companies', 'zero_companies', 'zero_percentage']
    sector_stats = sector_stats[sector_stats['total_companies'] >= 1]

    # Calculate weighted percentage
    total_zeros = sector_stats['zero_companies'].sum()
    sector_stats['weighted_percentage'] = (sector_stats['zero_companies'] / total_zeros * 100).round(1)

    # ANALYSIS 1: Weighted percentage of zero-reporting companies by sector

    # Bar plot for Analysis 1
    plt.figure(figsize=(12, 8))
    top_sectors_weighted = sector_stats.nlargest(15, 'weighted_percentage')
    bars1 = plt.bar(range(len(top_sectors_weighted)),
                    top_sectors_weighted['weighted_percentage'],
                    color='lightcoral', edgecolor='darkred', linewidth=0.5)

    plt.title('Sections Contributing Most to Zero-Reporting Companies\n(Weighted by Percentage of Total Zeros)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Percentage of Total Zero-Reporting Companies (%)', fontsize=12)
    plt.xlabel('Ateco Section Code', fontsize=12)

    # Use only Ateco codes as labels
    labels = [f"{idx}" for idx in top_sectors_weighted.index]
    plt.xticks(range(len(top_sectors_weighted)), labels, rotation=45, ha='right', fontsize=11)

    # Add value labels on bars
    for bar, value in zip(bars1, top_sectors_weighted['weighted_percentage']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('weighted_percentage_by_section.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Pie chart for Analysis 1
    plt.figure(figsize=(10, 8))

    # Group smaller sectors into "Others" for pie chart
    pie_data1 = sector_stats.nlargest(8, 'weighted_percentage').copy()
    others_sum1 = sector_stats['weighted_percentage'].sum() - pie_data1['weighted_percentage'].sum()
    others_row1 = pd.DataFrame({
        'weighted_percentage': [others_sum1],
        'zero_companies': [sector_stats['zero_companies'].sum() - pie_data1['zero_companies'].sum()]
    }, index=['Other'])

    pie_data1 = pd.concat([pie_data1, others_row1])

    # Create pie chart
    colors1 = plt.cm.Set3(np.linspace(0, 1, len(pie_data1)))
    wedges1, texts1, autotexts1 = plt.pie(pie_data1['weighted_percentage'],
                                          labels=pie_data1.index,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          colors=colors1,
                                          textprops={'fontsize': 11})

    plt.title('Distribution of Zero-Reporting Companies\nAcross Sections (Weighted)',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('section_distribution_weighted_pie.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # ANALYSIS 2: Percentage of companies with zeros within each sector

    # Bar plot for Analysis 2
    plt.figure(figsize=(12, 8))
    top_sectors_rate = sector_stats[sector_stats['total_companies'] >= 3].nlargest(15, 'zero_percentage')
    bars2 = plt.bar(range(len(top_sectors_rate)),
                    top_sectors_rate['zero_percentage'] * 100,
                    color='skyblue', edgecolor='navy', linewidth=0.5)

    plt.title('Sections with Highest Rate of Zero-Reporting Companies\n(Percentage within Section)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Percentage of Companies with Zero Reporting (%)', fontsize=12)
    plt.xlabel('Ateco Section Code', fontsize=12)
    plt.ylim(0, 110)

    # Use only Ateco codes as labels - increased bottom margin to prevent overlap
    labels2 = [f"{idx}" for idx in top_sectors_rate.index]
    plt.xticks(range(len(top_sectors_rate)), labels2, rotation=45, ha='right', fontsize=11)

    # Add value labels and sample size - adjusted positions to prevent overlap
    for i, (bar, value, total) in enumerate(
            zip(bars2, top_sectors_rate['zero_percentage'] * 100, top_sectors_rate['total_companies'])):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{value:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Move n= labels further down and make them smaller to prevent overlap with x-axis labels
        plt.text(bar.get_x() + bar.get_width() / 2, -8,
                 f'n={int(total)}', ha='center', va='top', fontsize=8, color='gray', alpha=0.8)

    # Adjust subplot parameters to give more room at the bottom
    plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()
    plt.savefig('section_zero_rates.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Pie chart for Analysis 2 - Showing more sectors without "Others" category
    plt.figure(figsize=(12, 10))

    # Get top sectors by zero rate (with at least 3 companies) - show more sectors
    top_pie_sectors = sector_stats[sector_stats['total_companies'] >= 3].nlargest(12, 'zero_percentage')

    # Create custom labels showing sector code, rate, and sample size
    def make_rate_label(idx, row):
        return f"Section {idx}\n{row['zero_percentage'] * 100:.0f}% (n={int(row['total_companies'])})"

    labels_pie2 = [make_rate_label(idx, row) for idx, row in top_pie_sectors.iterrows()]

    # Use the zero percentage as the size for the pie chart
    sizes = top_pie_sectors['zero_percentage'] * 100

    # Create a more distinct color palette
    colors2 = plt.cm.tab20(np.linspace(0, 1, len(top_pie_sectors)))

    # Print summary statistics
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total Companies: {len(df):,}")
    print(f"Zero-Reporting Companies: {df['all_zeros'].sum():,}")
    print(f"Overall Zero-Reporting Rate: {df['all_zeros'].mean() * 100:.1f}%")
    print(f"Total Sections (Ateco codes): {len(sector_stats)}")

    print(f"\nTop Section by Weighted Contribution:")
    top_weighted = sector_stats.nlargest(5, 'weighted_percentage')
    for idx, row in top_weighted.iterrows():
        print(f"  Section {idx}: {row['weighted_percentage']:.1f}% of total zeros")

    print(f"\nTop Sections by Zero-Reporting Rate (min. 3 companies):")
    top_rates = sector_stats[sector_stats['total_companies'] >= 3].nlargest(5, 'zero_percentage')
    for idx, row in top_rates.iterrows():
        print(f"  Section {idx}: {row['zero_percentage'] * 100:.1f}% zeros (n={int(row['total_companies'])})")

    # Additional statistics about high zero rate sectors
    print(f"\nSections with 100% Zero-Reporting (any sample size):")
    hundred_percent_sectors = sector_stats[sector_stats['zero_percentage'] == 1]
    for idx, row in hundred_percent_sectors.iterrows():
        print(f"  Section {idx}: {int(row['total_companies'])} companies")

def create_gri_to_esrs_sasb_analysis(df):
    """
    Create multiple visualizations for companies that had GRI in 2022/2023
    and now have SASB or ESRS in 2024
    """

    # Filter companies: GRI in 2022 OR 2023, AND (SASB OR ESRS in 2024)
    filtered_companies = df[
        ((df['GRI_2022'] == 1) | (df['GRI_2023'] == 1)) &
        ((df['SASB_2024'] == 1) | (df['ESRS_2024'] == 1))
        ].copy()

    print(f"Total companies meeting criteria: {len(filtered_companies)}")

    # Create adoption categories for 2024
    filtered_companies['Adoption_2024'] = 'None'
    filtered_companies.loc[
        (filtered_companies['ESRS_2024'] == 1) & (filtered_companies['SASB_2024'] == 0), 'Adoption_2024'
    ] = 'ESRS Only'
    filtered_companies.loc[
        (filtered_companies['SASB_2024'] == 1) & (filtered_companies['ESRS_2024'] == 0), 'Adoption_2024'
    ] = 'SASB Only'
    filtered_companies.loc[
        (filtered_companies['ESRS_2024'] == 1) & (filtered_companies['SASB_2024'] == 1), 'Adoption_2024'
    ] = 'Both'

    # ANALYSIS 1: 2-digit ATECO codes
    create_ateco_2digit_analysis(filtered_companies)

    # ANALYSIS 2: 1-character ATECO codes
    create_ateco_1char_analysis(filtered_companies)

    # ANALYSIS 3: Additional insights
    create_additional_insights(filtered_companies)
    return df



'''
THE NEXT 3 FUNCTIONS ARE ALREADY DEFAULT USED IN THE ONE BEFORE
'''
def create_ateco_2digit_analysis(df):
    """Create visualizations for 2-digit ATECO codes"""

    # Group by 2-digit ATECO and adoption type
    ateco_2digit_counts = df.groupby(['Ateco', 'Adoption_2024']).size().unstack(fill_value=0)

    # Calculate totals and sort
    ateco_2digit_counts['Total'] = ateco_2digit_counts.sum(axis=1)
    ateco_2digit_counts = ateco_2digit_counts.sort_values('Total', ascending=False)

    # Plot 1: Stacked bar chart for top 15 2-digit ATECO codes
    plt.figure(figsize=(15, 8))
    top_ateco_2digit = ateco_2digit_counts.head(15)

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # ESRS Only, SASB Only, Both

    bars = top_ateco_2digit[['ESRS Only', 'SASB Only', 'Both']].plot(
        kind='bar',
        stacked=True,
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )

    plt.title('Distribution of Companies with GRI (2022/2023) and ESRS/SASB (2024)\nby 2-Digit ATECO Code (Top 15)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('2-Digit ATECO Code', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Companies', fontsize=12, fontweight='bold')
    plt.legend(title='2024 Framework Adoption', title_fontsize=10, fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for container in bars.containers:
        bars.bar_label(container, label_type='center', fontsize=8, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('gri_to_esrs_sasb_2digit_stacked.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Plot 2: Percentage stacked bar chart
    plt.figure(figsize=(15, 8))

    # Calculate percentages
    top_ateco_percentage = top_ateco_2digit[['ESRS Only', 'SASB Only', 'Both']].div(
        top_ateco_2digit[['ESRS Only', 'SASB Only', 'Both']].sum(axis=1), axis=0
    ) * 100

    ax = top_ateco_percentage.plot(
        kind='bar',
        stacked=True,
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )

    plt.title(
        'Percentage Distribution of 2024 Framework Adoption\nfor Companies with GRI (2022/2023) by 2-Digit ATECO Code',
        fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('2-Digit ATECO Code', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.legend(title='2024 Framework Adoption', title_fontsize=10, fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig('gri_to_esrs_sasb_2digit_percentage.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Plot 3: Donut chart for overall adoption distribution
    overall_adoption = df['Adoption_2024'].value_counts()

    plt.figure(figsize=(10, 8))
    colors_donut = ['#2E86AB', '#A23B72', '#F18F01']

    wedges, texts, autotexts = plt.pie(
        overall_adoption.values,
        labels=overall_adoption.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_donut,
        pctdistance=0.85
    )

    # Draw a circle in the center to make it a donut
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gca().add_artist(centre_circle)

    plt.title('Overall 2024 Framework Adoption Distribution\nfor Companies with GRI (2022/2023)',
              fontsize=14, fontweight='bold', pad=20)

    # Improve text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    plt.tight_layout()
    plt.savefig('gri_to_esrs_sasb_overall_donut.png', dpi=300, bbox_inches='tight')
    #plt.show()
def create_ateco_1char_analysis(df):
    """Create visualizations for 1-character ATECO codes"""

    # Group by 1-character ATECO and adoption type
    ateco_1char_counts = df.groupby(['ateco', 'Adoption_2024']).size().unstack(fill_value=0)

    # Calculate totals and sort
    ateco_1char_counts['Total'] = ateco_1char_counts.sum(axis=1)
    ateco_1char_counts = ateco_1char_counts.sort_values('Total', ascending=False)

    # Plot 1: Grouped bar chart for 1-character ATECO
    plt.figure(figsize=(12, 8))

    x = np.arange(len(ateco_1char_counts))
    width = 0.25

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # ESRS Only, SASB Only, Both

    bars1 = plt.bar(x - width, ateco_1char_counts.get('ESRS Only', pd.Series([0] * len(ateco_1char_counts))),
                    width, label='ESRS Only', color=colors[0], edgecolor='white')
    bars2 = plt.bar(x, ateco_1char_counts.get('SASB Only', pd.Series([0] * len(ateco_1char_counts))),
                    width, label='SASB Only', color=colors[1], edgecolor='white')
    bars3 = plt.bar(x + width, ateco_1char_counts.get('Both', pd.Series([0] * len(ateco_1char_counts))),
                    width, label='Both', color=colors[2], edgecolor='white')

    plt.title('Distribution of Companies with GRI (2022/2023) and ESRS/SASB (2024)\nby 1-Character ATECO Section',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('ATECO Section', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Companies', fontsize=12, fontweight='bold')
    plt.xticks(x, ateco_1char_counts.index)
    plt.legend()
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('gri_to_esrs_sasb_1char_grouped.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Plot 2: Heatmap for 1-character ATECO
    plt.figure(figsize=(10, 6))

    # Prepare data for heatmap
    heatmap_data = ateco_1char_counts[['ESRS Only', 'SASB Only', 'Both']].fillna(0)

    sns.heatmap(
        heatmap_data.T,  # Transpose to have adoption types as rows
        annot=True,
        fmt='g',
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of Companies'},
        linewidths=0.5,
        linecolor='white'
    )

    plt.title('Heatmap: 2024 Framework Adoption by ATECO Section\nfor Companies with GRI (2022/2023)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('ATECO Section', fontsize=12, fontweight='bold')
    plt.ylabel('2024 Framework Adoption', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('gri_to_esrs_sasb_1char_heatmap.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Plot 3: Pie chart for 1-character ATECO distribution
    plt.figure(figsize=(12, 8))

    section_totals = ateco_1char_counts['Total']

    # Create pie chart
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(section_totals)))

    wedges, texts, autotexts = plt.pie(
        section_totals.values,
        labels=section_totals.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_pie
    )

    plt.title('Distribution of Companies with GRI (2022/2023) and ESRS/SASB (2024)\nby ATECO Section',
              fontsize=14, fontweight='bold', pad=20)

    # Improve text appearance
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig('gri_to_esrs_sasb_1char_pie.png', dpi=300, bbox_inches='tight')
    #plt.show()
def create_additional_insights(df):
    """Create additional insightful visualizations"""

    # Insight 1: Transition patterns from GRI years
    df['GRI_Pattern'] = 'No GRI'
    df.loc[(df['GRI_2022'] == 1) & (df['GRI_2023'] == 0), 'GRI_Pattern'] = 'GRI 2022 Only'
    df.loc[(df['GRI_2022'] == 0) & (df['GRI_2023'] == 1), 'GRI_Pattern'] = 'GRI 2023 Only'
    df.loc[(df['GRI_2022'] == 1) & (df['GRI_2023'] == 1), 'GRI_Pattern'] = 'GRI Both Years'

    # Cross-tabulation: GRI pattern vs 2024 adoption
    cross_tab = pd.crosstab(df['GRI_Pattern'], df['Adoption_2024'])

    # Plot: Stacked bar chart of GRI patterns vs 2024 adoption
    plt.figure(figsize=(12, 8))

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # ESRS Only, SASB Only, Both

    ax = cross_tab.plot(kind='bar', stacked=True, color=colors, figsize=(12, 8))

    plt.title('GRI Adoption Pattern (2022-2023) vs 2024 Framework Adoption',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('GRI Adoption Pattern', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Companies', fontsize=12, fontweight='bold')
    plt.legend(title='2024 Framework Adoption', title_fontsize=10, fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, label_type='center', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('gri_pattern_vs_2024_adoption.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # Insight 2: Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    total_companies = len(df)
    print(f"Total companies analyzed: {total_companies}")

    adoption_summary = df['Adoption_2024'].value_counts()
    print(f"\n2024 Framework Adoption:")
    for adoption_type, count in adoption_summary.items():
        percentage = (count / total_companies) * 100
        print(f"  {adoption_type}: {count} companies ({percentage:.1f}%)")

    gri_pattern_summary = df['GRI_Pattern'].value_counts()
    print(f"\nGRI Adoption Pattern (2022-2023):")
    for pattern, count in gri_pattern_summary.items():
        percentage = (count / total_companies) * 100
        print(f"  {pattern}: {count} companies ({percentage:.1f}%)")

    # Most common ATECO sections
    print(f"\nMost common ATECO sections (1-character):")
    ateco_section_counts = df['ateco'].value_counts().head(5)
    for section, count in ateco_section_counts.items():
        percentage = (count / total_companies) * 100
        print(f"  Section {section}: {count} companies ({percentage:.1f}%)")

    print(f"\nMost common 2-digit ATECO codes:")
    ateco_2digit_counts = df['Ateco'].value_counts().head(5)
    for code, count in ateco_2digit_counts.items():
        percentage = (count / total_companies) * 100
        print(f"  Code {code}: {count} companies ({percentage:.1f}%)")



#######################################

"""
#OUTDATED/EXECUTED PROCESSES
path1 = "Database Ufficiale.csv"       #Old naming for DF processing
df1= pd.read_csv(path1)


#Reshaping the database with GRI, ESRS, SASB and widening the columns of the previous indexes into the three years
# and creating a new cleaner csv and generating in csv and excel version
df = reshape_database(df1)


#Creating a table of piechart with the Usage or not of each index
pie_chart_per_year_per_standard_index(df)

#Computation of counts and percentages of the Industries with all 0s and 1s for each category throughout the 3 years
# and the computation of counts and percentage of Industries which presented either the SASB or the ESRS declaration in 2024
counts_and_percentages_and_law_check(df)

#Adding ateco identifiers
df_Extra = pd.read_csv('ATECO_codes.csv')
adding_new_Ateco_identifiers(df_Extra, df)

# Updating the csv and excel
df.to_csv('Starting_Dataset.csv.csv', index=False)
df.to_excel('Tidier_Dataset.xlsx', index=False)

#Reorder columns
print(df.columns.to_list())
df = df[['Name', 'ateco_section', 'ateco_sectionX', 'Ateco', 'AtecoX', 'ATECO', 'ATECOx', 'GRI_2022', 'ESRS_2022', 'SASB_2022', 'GRI_2023', 'ESRS_2023', 'SASB_2023', 'GRI_2024', 'ESRS_2024', 'SASB_2024']]
print(df.columns.to_list())

# Updating the csv and excel
df.to_csv('Starting_Dataset.csv.csv', index=False)
df.to_excel('Tidier_Dataset.xlsx', index=False)

#Second reshaping of the df
step1(df)
step2

figure1(df)
figure1_1(df)
figure2(df)
figure2_1(df)
figure3(df)
figure4(df)
figure5(df)

# Run the analysis of companies with 1 in GRI_2022 or GRI_2023 and 1 in either SASB_2024 or ESRS_2024 or both
create_gri_to_esrs_sasb_analysis(df)

START OF THE CODE
path = "Starting_Dataset.csv.csv"
df = pd.read_csv(path)
"""

#ACTUAL CODE
path = "Tidier_Dataset.csv"
df = pd.read_csv(path)

figure1(df)
figure1_1(df)
figure2(df)
figure2_1(df)
figure3(df)
figure4(df)
figure5(df)

create_gri_to_esrs_sasb_analysis(df)

# DEBUG


