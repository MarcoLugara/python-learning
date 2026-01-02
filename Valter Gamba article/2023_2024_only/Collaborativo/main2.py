import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import datetime
import os
import colorsys
from scipy import stats


def reshaping_database(input_file, output_file1, output_file2):
    """
    Remove all 2022 columns from the dataset file.

    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to save the output CSV file. If None, returns the DataFrame

    Returns:
    --------
    pd.DataFrame or None
        Returns the DataFrame if output_file is None, otherwise saves to file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Identify columns that end with '_2022'
    columns_to_keep = [col for col in df.columns if not col.endswith('_2022')]

    # Keep only non-2022 columns
    df = df[columns_to_keep]

    df.to_csv(output_file1, index=False)
    df.to_excel(output_file2, index=False)

    return df


def reshaping_database_part2(input_file, output_file1, output_file2):
    """
    Remove all 2022 columns from the dataset and rename fields with descriptive names.

    Parameters:
    -----------
    input_file : str
        The CSV content as a string

    Returns:
    --------
    str
        The reshaped CSV content without 2022 columns and with renamed field columns
    """
    # Read the CSV content
    df = pd.read_csv(input_file)

    # Define field name mapping
    field_mapping = {
        'Field1': 'Interviste',
        'Field2': 'Survey_Online',
        'Field3': 'Conference_Call',
        'Field4': 'Visite_In_Loco',
        'Field5': 'Seminari',
        'Field6': 'Conferenze',
        'Field7': 'Rapporti_Con_Agricoltori',
        'Field8': 'Roadshow'
    }

    # Create a dictionary to map old column names to new column names
    rename_dict = {}

    # For each column in the dataframe
    for col in df.columns:
        # Check if it's a field column (starts with 'Field' followed by a number)
        if col.startswith('Field') and '_' in col:
            # Extract field number and year
            field_part = col.split('_')[0]  # e.g., 'Field1'
            year = col.split('_')[1]  # e.g., '2022', '2023', '2024'

            # Skip 2022 columns
            if year == '2022':
                continue

            # If it's a field we have in our mapping, rename it
            if field_part in field_mapping:
                new_name = f"{field_mapping[field_part]}_{year}"
                rename_dict[col] = new_name
            else:
                # Keep as is if not in mapping (shouldn't happen with our data)
                rename_dict[col] = col
        else:
            # Keep non-field columns as they are
            rename_dict[col] = col

    # First, remove 2022 columns
    columns_to_keep = [col for col in df.columns if not col.endswith('_2022')]
    df = df[columns_to_keep]

    # Then rename the columns
    df = df.rename(columns=rename_dict)

    df.to_csv(output_file1, index=False)
    df.to_excel(output_file2, index=False)

    return df


"""
#OLD CODE
df = reshaping_database('Starting_Dataset.csv', 'Tidier_Dataset.csv', 'Tidier_Dataset.xlsx')
df = reshaping_database_part2('Tidier_Dataset.csv', 'Tidier_Dataset.csv', 'Tidier_Dataset.xlsx')
"""

# =============================================================================
# LOAD AND PREPARE DATA (MODIFIED FOR NEW DATASET)
# =============================================================================

# Load the data from the new CSV
df = pd.read_csv('Tidier_Dataset.csv')

# Extract ATECO letter from the 'ateco' column (it's already just a letter in this dataset)
df['ateco_letter'] = df['ateco']

# Define field groups - FIXED: Correct field names (8 separate fields)
# Based on the column names: Interviste, Survey_Online, Conference_Call, Visite_In_Loco,
# Seminari, Conferenze, Rapporti_Con_Agricoltori, Roadshow
field_groups = {
    'Single_Group': {
        'fields': ['Interviste', 'Survey_Online', 'Conference_Call', 'Visite_In_Loco',
                   'Seminari', 'Conferenze', 'Rapporti_Con_Agricoltori', 'Roadshow'],  # FIXED: 8 separate fields
        'color_base': '#4a90e2',  # Blue base color
        'name': 'All 8 Fields',
        'max_fields': 8
    }
}

# Calculate sum of ones for each group and year (only 2023 and 2024 in this dataset)
for group_name, group_info in field_groups.items():
    for year in ['2023', '2024']:
        field_columns = [f'{field}_{year}' for field in group_info['fields']]
        df[f'{group_name}_sum_{year}'] = df[field_columns].sum(axis=1)

# Calculate overall sum for each year
for year in ['2023', '2024']:
    all_field_columns = []
    for field in field_groups['Single_Group']['fields']:
        all_field_columns.append(f'{field}_{year}')
    df[f'total_sum_{year}'] = df[all_field_columns].sum(axis=1)


# =============================================================================
# COMPANY-LEVEL ANALYSIS FUNCTIONS (NEW ADDITION)
# =============================================================================

def analyze_company_performance(df):
    """
    Analyze company-level performance to find:
    1. Best performing companies in 2024 (highest sums, minimum 4/8)
    2. Best improvers between 2023 and 2024 (minimum improvement of 2 fields)

    Returns:
    --------
    dict: Dictionary with top performers and top improvers
    """
    # Ensure we have the necessary columns
    required_cols = ['Name', 'total_sum_2023', 'total_sum_2024']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe")
            return None

    # Create a copy to avoid modifying original
    company_analysis = df[['Name', 'ateco_letter', 'total_sum_2023', 'total_sum_2024']].copy()

    # Calculate improvement
    company_analysis['improvement'] = company_analysis['total_sum_2024'] - company_analysis['total_sum_2023']

    # Get top performers in 2024 (minimum 4/8)
    top_performers_2024 = company_analysis[
        company_analysis['total_sum_2024'] >= 4  # Only companies with at least 4/8
        ].sort_values(
        by=['total_sum_2024', 'Name'],
        ascending=[False, True]
    ).head(20)

    # Get top improvers (minimum improvement of 2 fields)
    top_improvers = company_analysis[
        company_analysis['improvement'] >= 2  # Only improvements of at least 2 fields
        ].sort_values(
        by=['improvement', 'total_sum_2024'],  # Sort by improvement, then by 2024 score
        ascending=[False, False]
    ).head(20)

    # Calculate statistics for summary
    total_companies = len(company_analysis)

    return {
        'top_performers_2024': top_performers_2024,
        'top_improvers': top_improvers,
        'summary_stats': {
            'total_companies_analyzed': total_companies,
            'companies_with_data_2024': len(company_analysis[company_analysis['total_sum_2024'] > 0]),
            'companies_min_4_2024': len(company_analysis[company_analysis['total_sum_2024'] >= 4]),
            'companies_improved_min_2': len(company_analysis[company_analysis['improvement'] >= 2]),
            'avg_2024_score': company_analysis['total_sum_2024'].mean(),
            'companies_perfect_2024': len(company_analysis[company_analysis['total_sum_2024'] == 8]),
            'companies_high_performers': len(top_performers_2024),
            'companies_significant_improvers': len(top_improvers)
        }
    }


# =============================================================================
# COLOR FUNCTIONS
# =============================================================================

def get_extreme_gradient(num_shades, reverse=False):
    """Generate a very distinct gradient from light yellow to dark red."""
    # Distinct color steps for better differentiation

    if num_shades == 8:
        # Pre-defined distinct colors for 8 shades
        colors = [
            '#ffffcc',  # Very light yellow (1 field)
            '#ffeda0',  # Light yellow (2 fields)
            '#fed976',  # Yellow-orange (3 fields)
            '#feb24c',  # Orange (4 fields)
            '#fd8d3c',  # Dark orange (5 fields)
            '#fc4e2a',  # Orange-red (6 fields)
            '#e31a1c',  # Red (7 fields)
            '#bd0026'  # Dark red (8 fields)
        ]
    else:
        # Fallback for other numbers of shades
        colors = []
        for i in range(num_shades):
            # Create a gradient from yellow to red
            hue = 0.1 - (i / (num_shades - 1)) * 0.1  # 0.1 (yellow) to 0.0 (red)
            saturation = 0.5 + (i / (num_shades - 1)) * 0.5  # 0.5 to 1.0
            lightness = 0.9 - (i / (num_shades - 1)) * 0.5  # 0.9 to 0.4
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}')

    if reverse:
        return colors[::-1]  # Reverse if needed
    return colors


# Generate color gradients for each group (excluding 0)
group_colors = {}
for group_name, group_info in field_groups.items():
    num_shades = group_info['max_fields']
    group_colors[group_name] = get_extreme_gradient(num_shades, reverse=True)

# =============================================================================
# PREPARE DATA FOR VISUALIZATION
# =============================================================================

# Create detailed breakdown by ATECO letter and sum values for each group
detailed_breakdown = pd.DataFrame()

for year in ['2023', '2024']:  # MODIFIED: Only 2023 and 2024
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
# ENHANCED STATISTICAL ANALYSIS (MODIFIED FOR 2 YEARS)
# =============================================================================

def calculate_comprehensive_metrics():
    """Calculate comprehensive statistical metrics for each sector."""

    metrics_list = []

    for sector in sector_order:
        sector_data = df[df['ateco_letter'] == sector]
        total_companies = len(sector_data)

        sector_metrics = {
            'Sector': sector,
            'Total_Companies': total_companies
        }

        # Year-wise metrics (only 2023 and 2024)
        for year in ['2023', '2024']:
            # Overall completion metrics
            total_fields_completed = sector_data[f'total_sum_{year}'].sum()
            max_possible = 8 * total_companies  # MODIFIED: 8 fields instead of 16
            overall_completion = (total_fields_completed / max_possible * 100) if max_possible > 0 else 0

            sector_metrics[f'overall_completion_{year}'] = overall_completion
            sector_metrics[
                f'avg_fields_per_company_{year}'] = total_fields_completed / total_companies if total_companies > 0 else 0

            # Percentage with any completion
            companies_with_any = len(sector_data[sector_data[f'total_sum_{year}'] > 0])
            sector_metrics[f'pct_with_any_{year}'] = (
                    companies_with_any / total_companies * 100) if total_companies > 0 else 0

            # Percentage with full completion (8 fields) - MODIFIED
            companies_with_full = len(sector_data[sector_data[f'total_sum_{year}'] == 8])
            sector_metrics[f'pct_with_full_{year}'] = (
                    companies_with_full / total_companies * 100) if total_companies > 0 else 0

            # Group-specific metrics (only one group in this dataset)
            for group_name, group_info in field_groups.items():
                sum_col = f'{group_name}_sum_{year}'
                max_fields = group_info['max_fields']

                # Group completion rate
                group_fields_completed = sector_data[sum_col].sum()
                group_max_possible = max_fields * total_companies
                group_completion = (group_fields_completed / group_max_possible * 100) if group_max_possible > 0 else 0

                # Percentage with any in group
                companies_with_any_group = len(sector_data[sector_data[sum_col] > 0])
                pct_any_group = (companies_with_any_group / total_companies * 100) if total_companies > 0 else 0

                # Percentage with full group completion
                companies_with_full_group = len(sector_data[sector_data[sum_col] == max_fields])
                pct_full_group = (companies_with_full_group / total_companies * 100) if total_companies > 0 else 0

                sector_metrics[f'{group_name}_completion_{year}'] = group_completion
                sector_metrics[f'{group_name}_pct_any_{year}'] = pct_any_group
                sector_metrics[f'{group_name}_pct_full_{year}'] = pct_full_group

        # Calculate trends (2023 to 2024 instead of 2022 to 2024)
        sector_metrics['trend_overall_23_24'] = sector_metrics.get('overall_completion_2024', 0) - sector_metrics.get(
            'overall_completion_2023', 0)
        sector_metrics['trend_any_23_24'] = sector_metrics.get('pct_with_any_2024', 0) - sector_metrics.get(
            'pct_with_any_2023', 0)

        # Calculate averages across years
        for metric in ['overall_completion', 'pct_with_any', 'pct_with_full']:
            values = [sector_metrics.get(f'{metric}_{year}', 0) for year in ['2023', '2024']]
            sector_metrics[f'avg_{metric}'] = np.mean(values)

        for group_name in field_groups.keys():
            completion_values = [sector_metrics.get(f'{group_name}_completion_{year}', 0) for year in
                                 ['2023', '2024']]
            any_values = [sector_metrics.get(f'{group_name}_pct_any_{year}', 0) for year in ['2023', '2024']]

            sector_metrics[f'avg_{group_name}_completion'] = np.mean(completion_values)
            sector_metrics[f'avg_{group_name}_pct_any'] = np.mean(any_values)

            # Group-specific trends
            sector_metrics[f'trend_{group_name}_completion_23_24'] = (
                    sector_metrics.get(f'{group_name}_completion_2024', 0) -
                    sector_metrics.get(f'{group_name}_completion_2023', 0)
            )

        metrics_list.append(sector_metrics)

    return pd.DataFrame(metrics_list)


# =============================================================================
# ADVANCED STATISTICAL ANALYSIS (MODIFIED FOR 2 YEARS)
# =============================================================================

def calculate_advanced_statistics():
    """Calculate advanced statistical measures."""

    stats_dict = {}

    # Overall statistics
    for year in ['2023', '2024']:
        year_stats = {
            'total_companies': len(df),
            'companies_with_any': len(df[df[f'total_sum_{year}'] > 0]),
            'companies_with_full': len(df[df[f'total_sum_{year}'] == 8]),  # MODIFIED: 8 fields
            'total_fields_completed': df[f'total_sum_{year}'].sum(),
            'max_possible_fields': len(df) * 8  # MODIFIED: 8 fields
        }

        year_stats['pct_with_any'] = (year_stats['companies_with_any'] / year_stats['total_companies'] * 100) if \
            year_stats['total_companies'] > 0 else 0
        year_stats['pct_with_full'] = (year_stats['companies_with_full'] / year_stats['total_companies'] * 100) if \
            year_stats['total_companies'] > 0 else 0
        year_stats['overall_completion'] = (
                year_stats['total_fields_completed'] / year_stats['max_possible_fields'] * 100) if year_stats[
                                                                                                       'max_possible_fields'] > 0 else 0
        year_stats['avg_fields_per_company'] = df[f'total_sum_{year}'].mean()

        # Distribution statistics
        year_stats['median_fields'] = df[f'total_sum_{year}'].median()
        year_stats['std_fields'] = df[f'total_sum_{year}'].std()
        year_stats['q1_fields'] = df[f'total_sum_{year}'].quantile(0.25)
        year_stats['q3_fields'] = df[f'total_sum_{year}'].quantile(0.75)

        stats_dict[year] = year_stats

    # Engagement distribution analysis
    distribution_data = {}
    for year in ['2023', '2024']:
        distribution = []
        total_companies = len(df)
        for count in range(0, 9):  # MODIFIED: 0 to 8 fields
            companies = len(df[df[f'total_sum_{year}'] == count])
            percentage = (companies / total_companies * 100) if total_companies > 0 else 0
            distribution.append({
                'fields': count,
                'companies': companies,
                'percentage': percentage
            })
        distribution_data[year] = distribution

    # Trend analysis (2023 to 2024)
    trend_stats = {
        'overall_completion_growth': stats_dict['2024']['overall_completion'] - stats_dict['2023'][
            'overall_completion'],
        'pct_with_any_growth': stats_dict['2024']['pct_with_any'] - stats_dict['2023']['pct_with_any'],
        'avg_fields_growth': stats_dict['2024']['avg_fields_per_company'] - stats_dict['2023']['avg_fields_per_company']
    }

    # Sector correlation analysis
    sector_correlations = {}
    performance_data = calculate_comprehensive_metrics()

    # Correlation between sector size and performance
    if len(performance_data) > 1:
        sector_size = performance_data['Total_Companies']
        avg_performance = performance_data['avg_overall_completion']

        # Remove NaN values
        valid_idx = ~sector_size.isna() & ~avg_performance.isna()
        if valid_idx.sum() > 1:
            try:
                corr_coefficient, p_value = stats.pearsonr(
                    sector_size[valid_idx],
                    avg_performance[valid_idx]
                )
                sector_correlations['size_vs_performance'] = {
                    'correlation': corr_coefficient,
                    'p_value': p_value,
                    'interpretation': 'Positive correlation suggests larger sectors perform better' if corr_coefficient > 0 else 'Negative correlation suggests larger sectors perform worse'
                }
            except:
                sector_correlations['size_vs_performance'] = {
                    'correlation': 0,
                    'p_value': 1,
                    'interpretation': 'Unable to calculate correlation'
                }

    return {
        'yearly_stats': stats_dict,
        'distribution_data': distribution_data,
        'trend_stats': trend_stats,
        'sector_correlations': sector_correlations
    }


# =============================================================================
# PERFORMANCE CLASSIFICATION
# =============================================================================

def classify_sectors(performance_df):
    """Classify sectors based on multiple criteria."""

    classifications = {}

    # Overall performance classification (adjusted thresholds for 8 fields)
    if 'avg_overall_completion' in performance_df.columns:
        overall_avg = performance_df['avg_overall_completion']
        classifications['high_performers'] = performance_df[overall_avg > 50] if len(
            performance_df) > 0 else pd.DataFrame()  # Adjusted threshold
        classifications['medium_performers'] = performance_df[(overall_avg >= 25) & (overall_avg <= 50)] if len(
            performance_df) > 0 else pd.DataFrame()  # Adjusted threshold
        classifications['low_performers'] = performance_df[overall_avg < 25] if len(
            performance_df) > 0 else pd.DataFrame()  # Adjusted threshold

    # Engagement level classification
    if 'avg_pct_with_any' in performance_df.columns:
        engagement_avg = performance_df['avg_pct_with_any']
        classifications['highly_engaged'] = performance_df[engagement_avg > 60] if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['moderately_engaged'] = performance_df[(engagement_avg >= 30) & (engagement_avg <= 60)] if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['low_engaged'] = performance_df[engagement_avg < 30] if len(
            performance_df) > 0 else pd.DataFrame()

    # Trend classification (2023 to 2024)
    if 'trend_overall_23_24' in performance_df.columns:
        trend = performance_df['trend_overall_23_24']
        classifications['biggest_improvers'] = performance_df.nlargest(min(5, len(performance_df)),
                                                                       'trend_overall_23_24') if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['biggest_decliners'] = performance_df.nsmallest(min(5, len(performance_df)),
                                                                        'trend_overall_23_24') if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['consistent_performers'] = performance_df[abs(trend) < 5] if len(
            performance_df) > 0 else pd.DataFrame()

    # Group leaders (only one group in this dataset)
    for group_name in field_groups.keys():
        avg_col = f'avg_{group_name}_completion'
        if avg_col in performance_df.columns:
            classifications[f'{group_name}_leaders'] = performance_df.nlargest(min(5, len(performance_df)),
                                                                               avg_col) if len(
                performance_df) > 0 else pd.DataFrame()
            classifications[f'{group_name}_laggards'] = performance_df.nsmallest(min(5, len(performance_df)),
                                                                                 avg_col) if len(
                performance_df) > 0 else pd.DataFrame()

    # Size-based analysis
    if 'Total_Companies' in performance_df.columns and 'avg_overall_completion' in performance_df.columns:
        classifications['large_high_performers'] = performance_df[
            (performance_df['Total_Companies'] > performance_df['Total_Companies'].median()) &
            (performance_df['avg_overall_completion'] > 30)
            ] if len(performance_df) > 0 else pd.DataFrame()

        classifications['small_excellent'] = performance_df[
            (performance_df['Total_Companies'] <= 5) &
            (performance_df['avg_overall_completion'] > 50)
            ].sort_values('avg_overall_completion', ascending=False) if len(performance_df) > 0 else pd.DataFrame()

    return classifications


# =============================================================================
# CREATE VISUALIZATIONS - ABSOLUTE VALUES (ADAPTED FOR SINGLE GROUP)
# =============================================================================

def create_absolute_bar_chart(year_data, year, sector_order):
    """Create a bar chart with absolute counts (excluding zero values)."""
    fig, ax = plt.subplots(figsize=(16, 8))

    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.18, top=0.92)

    # Set up bar positions and width
    x_positions = np.arange(len(sector_order))
    bar_width = 0.7

    # Prepare data structure: sector -> group -> value -> count
    sector_data = {}
    sector_totals = {}
    for sector in sector_order:
        sector_row = year_data[year_data['ateco_letter'] == sector]
        if not sector_row.empty:
            total_companies = sector_row.iloc[0]['total_companies']
            sector_totals[sector] = total_companies
            sector_dict = {}
            for group_name in field_groups.keys():
                group_dict = {}
                max_fields = field_groups[group_name]['max_fields']
                for value in range(1, max_fields + 1):  # Start from 1, excluding 0
                    count = sector_row.iloc[0][f'{group_name}_{value}']
                    group_dict[value] = count
                sector_dict[group_name] = group_dict
            sector_data[sector] = sector_dict

    # Create the stacked bars (only one group in this dataset)
    group_order = ['Single_Group']

    # Initialize bottoms for each bar
    bottoms = np.zeros(len(sector_order))

    # Keep track of which colors we've added to legend
    legend_added = {}

    # Stack from 1 to max_fields (lightest to darkest)
    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Stack from 1 to max_fields (lightest to darkest)
        for value in range(1, max_fields + 1):
            # Get counts for this value across all sectors
            counts = []
            for i, sector in enumerate(sector_order):
                if sector in sector_data:
                    counts.append(sector_data[sector][group_name].get(value, 0))
                else:
                    counts.append(0)

            if sum(counts) > 0:  # Only add if there are any
                color_idx = value - 1
                color = colors_for_group[color_idx]

                bars = ax.bar(x_positions, counts, bottom=bottoms,
                              width=bar_width, color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
                bottoms += counts

                # Add to legend if not already added
                legend_key = f"{group_name}_{value}"
                if legend_key not in legend_added:
                    legend_added[legend_key] = (color, value)

    # Calculate total heights for sorting
    total_heights = bottoms

    # Sort sectors by total height (descending)
    sorted_indices = np.argsort(total_heights)[::-1]
    sorted_sectors = [sector_order[i] for i in sorted_indices]

    # Recreate the figure with sorted data
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.18, top=0.92)

    x_positions_sorted = np.arange(len(sorted_sectors))
    bottoms_sorted = np.zeros(len(sorted_sectors))

    # Recreate sorted sector data
    sorted_sector_data = {}
    sorted_sector_totals = {}
    for sector in sorted_sectors:
        if sector in sector_data:
            sorted_sector_data[sector] = sector_data[sector]
            sorted_sector_totals[sector] = sector_totals[sector]

    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        for value in range(1, max_fields + 1):
            counts = []
            for sector in sorted_sectors:
                if sector in sorted_sector_data:
                    counts.append(sorted_sector_data[sector][group_name].get(value, 0))
                else:
                    counts.append(0)

            if sum(counts) > 0:
                color_idx = value - 1
                color = colors_for_group[color_idx]

                ax.bar(x_positions_sorted, counts, bottom=bottoms_sorted,
                       width=bar_width, color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
                bottoms_sorted += counts

    # Customize the plot
    ax.set_title(f'{year} - Field Completion Distribution (Absolute Counts)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ATECO Sector', fontsize=12)
    ax.set_ylabel('Number of Companies', fontsize=12)

    # Set x-ticks with better positioning
    ax.set_xticks(x_positions_sorted)
    ax.set_xticklabels(sorted_sectors, rotation=45, ha='right', fontsize=12)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total count labels on top of bars
    max_height = max(bottoms_sorted) if len(bottoms_sorted) > 0 else 1
    for i, total in enumerate(bottoms_sorted):
        if total > 0:
            ax.text(i, total + max_height * 0.01, f'{int(total)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # NEW: Adjust y-axis to bring highest bar close to top
    # Get current y-axis limits
    y_min, y_max_current = ax.get_ylim()

    # Calculate what the new y_max should be
    # We want the highest bar to be close to the top but not touching
    # Let's set it so the highest bar reaches about 90% of the y-axis height
    if max_height > 0:
        # Calculate padding: about 10% above the highest bar
        padding = max_height * 0.1
        new_y_max = max_height + padding

        # Make sure we have at least some minimum range
        if new_y_max - y_min < max_height * 0.2:
            new_y_max = max_height * 1.2
    else:
        # Fallback if all bars are 0
        new_y_max = 10

    # Set the new y-axis limits
    ax.set_ylim(bottom=y_min, top=new_y_max)

    # Recalculate offset for 'n' labels based on new y-axis
    offset = new_y_max * 0.05

    # Add 'n' labels at the bottom
    ax.set_ylim(bottom=y_min - offset)
    n_label_y_position = y_min - offset * 0.3

    # Add 'n' labels
    for i, sector in enumerate(sorted_sectors):
        total_companies = sorted_sector_totals.get(sector, 0)
        ax.text(i, n_label_y_position, f'n={total_companies}',
                ha='center', va='top', fontsize=9, color='gray', fontweight='bold')

    # Create a comprehensive legend
    legend_items = []
    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Add group header
        legend_items.append((Line2D([0], [0], color='white'), f"{group_info['name']}:"))

        # Add color items for this group
        for value in range(1, max_fields + 1):
            color_idx = value - 1
            if f"{group_name}_{value}" in legend_added:
                color = colors_for_group[color_idx]
                legend_items.append((Patch(facecolor=color, edgecolor='black', alpha=0.9, linewidth=0.5),
                                     f"  {value} field{'s' if value > 1 else ''}"))

    # Create legend
    if legend_items:
        legend_handles, legend_labels = zip(*legend_items)
        ax.legend(legend_handles, legend_labels,
                  title='Completed Fields',
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=10,
                  title_fontsize=11,
                  frameon=True,
                  fancybox=True,
                  shadow=True)

    return fig, sorted_sectors


# =============================================================================
# CREATE VISUALIZATIONS - PERCENTAGES (ADAPTED FOR SINGLE GROUP)
# =============================================================================

def create_percentage_bar_chart(year_data, year, sector_order):
    """Create a stacked bar chart showing percentage of total fields completed."""
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.18, top=0.92)

    # Calculate percentages for each sector
    sector_stratified_data = {}
    sector_totals = {}
    sector_total_percentages = {}

    for sector in sector_order:
        sector_row = year_data[year_data['ateco_letter'] == sector]
        if not sector_row.empty:
            total_companies = sector_row.iloc[0]['total_companies']
            sector_totals[sector] = total_companies

            # Calculate total possible fields (8 fields per company) - MODIFIED
            total_possible_fields = 8 * total_companies if total_companies > 0 else 1

            # For each group and value (1 to max_fields), calculate contribution percentage
            stratified_dict = {}
            total_percentage = 0

            for group_name, group_info in field_groups.items():
                group_dict = {}
                max_fields = group_info['max_fields']

                for value in range(1, max_fields + 1):
                    count = sector_row.iloc[0].get(f'{group_name}_{value}', 0)
                    fields_completed = value * count
                    percentage = (fields_completed / total_possible_fields * 100) if total_possible_fields > 0 else 0
                    group_dict[value] = percentage
                    total_percentage += percentage

                stratified_dict[group_name] = group_dict

            sector_stratified_data[sector] = stratified_dict
            sector_total_percentages[sector] = total_percentage

    # Sort sectors by total percentage (descending)
    sorted_sectors = sorted(sector_total_percentages.items(), key=lambda x: x[1], reverse=True)
    sorted_sector_names = [sector for sector, _ in sorted_sectors]

    # Create the stacked bars
    group_order = ['Single_Group']

    # Initialize bottoms for each bar
    x_positions_sorted = np.arange(len(sorted_sector_names))
    bottoms = np.zeros(len(sorted_sector_names))

    # Keep track of which colors/values we've added to legend
    legend_added = {}

    # Stack in the same order
    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Stack from 1 to max_fields (lightest to darkest)
        for value in range(1, max_fields + 1):
            percentages = []
            for sector in sorted_sector_names:
                if sector in sector_stratified_data:
                    percentages.append(sector_stratified_data[sector][group_name].get(value, 0))
                else:
                    percentages.append(0)

            if sum(percentages) > 0:
                color_idx = value - 1
                color = colors_for_group[color_idx] if color_idx < len(colors_for_group) else colors_for_group[-1]

                ax.bar(x_positions_sorted, percentages, bottom=bottoms,
                       width=0.7, color=color, alpha=0.9,
                       edgecolor='black', linewidth=0.5)
                bottoms += percentages

                legend_key = f"{group_name}_{value}"
                if legend_key not in legend_added:
                    legend_added[legend_key] = (color, group_name, value)

    # Customize the plot
    ax.set_title(f'{year} - Percentage of Total Fields Completed',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ATECO Sector', fontsize=12)
    ax.set_ylabel('Percentage of Total Possible Fields Completed (%)', fontsize=12)

    ax.set_xticks(x_positions_sorted)
    ax.set_xticklabels(sorted_sector_names, rotation=45, ha='right', fontsize=12)

    # Set y-axis limit
    max_bar_height = max(bottoms) if len(bottoms) > 0 else 0
    y_max = min(100, max_bar_height * 1.1)
    if y_max < 10:
        y_max = 10

    y_min = 0
    offset = y_max * 0.05
    ax.set_ylim(bottom=y_min - offset, top=y_max)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total percentage labels on top of bars
    for i, (sector, total_pct) in enumerate(zip(sorted_sector_names, bottoms)):
        if total_pct > 0:
            ax.text(i, total_pct + y_max * 0.01, f'{int(total_pct)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add 'n' labels at the bottom
    n_label_y_position = y_min - offset * 0.3

    for i, sector in enumerate(sorted_sector_names):
        total_companies = sector_totals.get(sector, 0)
        ax.text(i, n_label_y_position, f'n={total_companies}',
                ha='center', va='top', fontsize=9, color='gray', fontweight='bold')

    # Create legend
    legend_items = []
    for group_name in group_order:
        group_info = field_groups[group_name]
        legend_items.append((Line2D([0], [0], color='white'),
                             f"{group_info['name']}:"))

        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        for value in range(1, max_fields + 1):
            legend_key = f"{group_name}_{value}"
            if legend_key in legend_added:
                color, _, val = legend_added[legend_key]
                legend_items.append((Patch(facecolor=color, edgecolor='black',
                                           alpha=0.9, linewidth=0.5),
                                     f"  {val} field{'s' if val > 1 else ''}"))

    if legend_items:
        legend_handles, legend_labels = zip(*legend_items)
        ax.legend(legend_handles, legend_labels,
                  title='Completed Fields',
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=10,
                  title_fontsize=11,
                  frameon=True,
                  fancybox=True,
                  shadow=True)

    return fig, sorted_sector_names


# =============================================================================
# CREATE ALL VISUALIZATIONS (MODIFIED FOR 2 YEARS)
# =============================================================================

# Create visualizations for each year
image_paths_absolute = []
image_paths_percentage = []
all_sorted_sectors = {}

print("  - Creating absolute value charts...")
for year in ['2023', '2024']:  # MODIFIED: Only 2023 and 2024
    try:
        year_data = detailed_breakdown[detailed_breakdown['year'] == year]
        fig_abs, sorted_sectors = create_absolute_bar_chart(year_data, year, sector_order)
        abs_path = f'Image_{year}_Absolute.png'
        plt.savefig(abs_path, dpi=300, bbox_inches='tight')
        image_paths_absolute.append(abs_path)
        all_sorted_sectors[f'{year}_abs'] = sorted_sectors
        plt.close()
        print(f"    ✓ Created {abs_path}")
    except Exception as e:
        print(f"    ✗ Error creating absolute chart for {year}: {str(e)[:100]}...")
        image_paths_absolute.append(None)

print("  - Creating percentage charts...")
for year in ['2023', '2024']:  # MODIFIED: Only 2023 and 2024
    try:
        year_data = detailed_breakdown[detailed_breakdown['year'] == year]
        fig_pct, sorted_sectors = create_percentage_bar_chart(year_data, year, sector_order)
        pct_path = f'Image_{year}_Percentage.png'
        plt.savefig(pct_path, dpi=300, bbox_inches='tight')
        image_paths_percentage.append(pct_path)
        all_sorted_sectors[f'{year}_pct'] = sorted_sectors
        plt.close()
        print(f"    ✓ Created {pct_path}")
    except Exception as e:
        print(f"    ✗ Error creating percentage chart for {year}: {str(e)[:100]}...")
        image_paths_percentage.append(None)

# =============================================================================
# PERFORM COMPREHENSIVE ANALYSIS
# =============================================================================

print("  - Calculating performance metrics...")
performance_analysis = calculate_comprehensive_metrics()

print("  - Calculating advanced statistics...")
advanced_stats = calculate_advanced_statistics()

print("  - Classifying sectors...")
sector_classifications = classify_sectors(performance_analysis)

# Analyze company performance (NEW)
print("  - Analyzing company-level performance...")
company_analysis = analyze_company_performance(df)

# Prepare summary table for PDF
summary_data = performance_analysis[['Sector', 'Total_Companies',
                                     'overall_completion_2024',
                                     'avg_fields_per_company_2024',
                                     'pct_with_any_2024',
                                     'trend_overall_23_24']].copy()
summary_data = summary_data.sort_values('overall_completion_2024', ascending=False)
summary_data.columns = ['Sector', 'Total Cos', '2024: Overall %', '2024: Avg Fields', '2024: % Any', 'Trend 23-24']
summary_table_data = summary_data.round(1).values.tolist()
summary_table_headers = ['Sector', 'Total Cos', '2024: Overall %', '2024: Avg Fields', '2024: % Any', 'Trend 23-24']


# =============================================================================
# ENHANCED PDF REPORT (ADAPTED FOR NEW DATASET)
# =============================================================================

def create_pdf_report():
    from reportlab.lib import colors

    # Create PDF document
    pdf_filename = f"Analysis_Report_Enhanced_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=landscape(letter))

    # Get styles
    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#2c3e50'),
        alignment=TA_CENTER,
        spaceAfter=25
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

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=normal_style,
        fontSize=9,
        alignment=TA_CENTER,
        leading=11
    )

    # Create story (content) list
    story = []

    # Title page
    story.append(Paragraph("ATECO Sector Analysis - Enhanced Statistical Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Analysis Date: {datetime.datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph("Dataset: 8 fields, 2 years (2023-2024)", styles['Normal']))
    story.append(Paragraph("Field Group: Interviste, Survey_Online, Conference_Call, Visite_In_Loco, "
                           "Seminari, Conferenze, Rapporti_Con_Agricoltori, Roadshow", styles['Normal']))
    story.append(Spacer(1, 20))

    # Executive Summary with Statistical Insights
    story.append(Paragraph("Executive Summary with Statistical Insights", heading1_style))

    # Key statistical findings
    key_stats = advanced_stats['yearly_stats']['2024']
    trend_stats = advanced_stats['trend_stats']

    key_data = [
        ['Statistical Metric', 'Value'],
        ['Total Companies Analyzed', f"{key_stats['total_companies']:,}"],
        ['Total ATECO Sectors', f"{len(sector_order)}"],
        ['Overall Completion Rate (2024)', f"{key_stats['overall_completion']:.1f}%"],
        ['Companies with Any Data (2024)', f"{key_stats['pct_with_any']:.1f}%"],
        ['Companies with All 8 Fields (2024)', f"{key_stats['pct_with_full']:.1f}%"],
        ['Average Fields per Company (2024)', f"{key_stats['avg_fields_per_company']:.2f}"],
        ['Median Fields per Company (2024)', f"{key_stats['median_fields']:.1f}"],
        ['Standard Deviation (2024)', f"{key_stats['std_fields']:.2f}"],
        ['Overall Trend 2023-2024', f"{trend_stats['overall_completion_growth']:+.1f}%"],
        ['Engagement Growth 2023-2024', f"{trend_stats['pct_with_any_growth']:+.1f}%"],
        ['High Performing Sectors (>50%)', f"{len(sector_classifications.get('high_performers', pd.DataFrame()))}"],
        ['Medium Performing Sectors (25-50%)',
         f"{len(sector_classifications.get('medium_performers', pd.DataFrame()))}"],
        ['Low Performing Sectors (<25%)', f"{len(sector_classifications.get('low_performers', pd.DataFrame()))}"]
    ]

    key_table = Table(key_data, colWidths=[240, 160])
    key_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]))
    story.append(key_table)
    story.append(PageBreak())

    # Section 1: Distribution Analysis
    story.append(Paragraph("1. Statistical Distribution Analysis", heading1_style))
    story.append(
        Paragraph("This section shows how many companies completed different numbers of fields (0-8).", normal_style))

    # Distribution statistics table
    dist_headers = ['Fields', 'Companies (2023)', '%', 'Companies (2024)', '%']
    dist_data = [dist_headers]

    # Show distributions for 0-8 fields
    ranges = [
        (0, 0, "0"),
        (1, 2, "1-2"),
        (3, 4, "3-4"),
        (5, 6, "5-6"),
        (7, 8, "7-8")
    ]

    for start, end, label in ranges:
        row = [label]
        for year in ['2023', '2024']:
            year_dist = advanced_stats['distribution_data'][year]
            if start == end:
                if start < len(year_dist):
                    companies = year_dist[start]['companies']
                    percentage = year_dist[start]['percentage']
                    row.append(f"{companies:,}")
                    row.append(f"{percentage:.1f}%")
                else:
                    row.append("0")
                    row.append("0.0%")
            else:
                total_companies = 0
                total_pct = 0
                for i in range(start, end + 1):
                    if i < len(year_dist):
                        total_companies += year_dist[i]['companies']
                        total_pct += year_dist[i]['percentage']
                row.append(f"{total_companies:,}")
                row.append(f"{total_pct:.1f}%")
        dist_data.append(row)

    dist_table = Table(dist_data, colWidths=[90, 90, 80, 90, 80])
    dist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 1), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
    ]))
    story.append(dist_table)
    story.append(Spacer(1, 15))

    # Add statistical summary of distribution
    story.append(Paragraph("Key Points:", heading2_style))

    # Calculate mode safely
    mode_values = df['total_sum_2024'].mode()
    mode_str = f"{mode_values.iloc[0]}" if not mode_values.empty else 'N/A'

    story.append(Paragraph(f"• Most common: {mode_str} fields completed (most frequent value)", bullet_style))
    story.append(
        Paragraph(f"• 25% of companies complete ≤{advanced_stats['yearly_stats']['2024']['q1_fields']:.1f} fields",
                  bullet_style))
    story.append(Paragraph(
        f"• 50% of companies complete ≤{advanced_stats['yearly_stats']['2024']['median_fields']:.1f} fields (median)",
        bullet_style))
    story.append(
        Paragraph(f"• 75% of companies complete ≤{advanced_stats['yearly_stats']['2024']['q3_fields']:.1f} fields",
                  bullet_style))

    story.append(PageBreak())

    # Section 2: Absolute Value Visualizations
    story.append(Paragraph("2. Absolute Value Charts by Year", heading1_style))
    story.append(Paragraph(
        "Each bar shows the number of companies with completed fields in each sector. Sorted by total count (descending).",
        normal_style))
    story.append(Paragraph("How to read: Taller bars = more companies with completed fields", bullet_style))
    story.append(Paragraph("Color meaning: Different colors show different numbers of completed fields (see legend)",
                           bullet_style))
    story.append(Paragraph("n= shows total number of companies in that sector", bullet_style))

    for year, image_path in zip(['2023', '2024'], image_paths_absolute):
        if image_path and os.path.exists(image_path):
            content_to_keep_together = []
            content_to_keep_together.append(Paragraph(f"{year} - Absolute Counts (Sorted by Total)", heading2_style))
            try:
                content_to_keep_together.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                content_to_keep_together.append(Paragraph("Image not available", normal_style))
            content_to_keep_together.append(Spacer(1, 5))
            story.append(KeepTogether(content_to_keep_together))

    story.append(PageBreak())

    # Section 3: Percentage Visualizations
    story.append(Paragraph("3. Percentage Charts by Year", heading1_style))
    story.append(Paragraph(
        "Each bar shows what percentage of the total possible fields were completed in each sector. Sorted by percentage (descending).",
        normal_style))
    story.append(Paragraph("How to read: Taller bars = higher percentage of fields completed", bullet_style))
    story.append(Paragraph("Note: 100% would mean ALL companies completed ALL 8 fields", bullet_style))
    story.append(Paragraph("Color meaning: Same as absolute charts (see legend)", bullet_style))
    story.append(Paragraph("n= shows total number of companies in that sector", bullet_style))

    for year, image_path in zip(['2023', '2024'], image_paths_percentage):
        if image_path and os.path.exists(image_path):
            content_to_keep_together = []
            content_to_keep_together.append(
                Paragraph(f"{year} - Percentage Distribution (Sorted by %)", heading2_style))
            try:
                content_to_keep_together.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                content_to_keep_together.append(Paragraph("Image not available", normal_style))
            content_to_keep_together.append(Spacer(1, 5))
            story.append(KeepTogether(content_to_keep_together))

    story.append(PageBreak())

    # Section 4: Sector Performance Classification
    story.append(Paragraph("4. Sector Performance Classification", heading1_style))
    story.append(Paragraph("Note: Classification based on average performance across 2023-2024", normal_style))
    story.append(Spacer(1, 10))

    # High Performers
    high_performers = sector_classifications.get('high_performers', pd.DataFrame())
    if len(high_performers) > 0:
        story.append(
            Paragraph(f"High Performers (>50% overall completion): {len(high_performers)} sectors", heading2_style))
        story.append(Paragraph("These sectors consistently completed the highest percentage of fields:", normal_style))
        for _, row in high_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg completion, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend (2023-2024): {row.get('trend_overall_23_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No high performing sectors found", normal_style))

    story.append(Spacer(1, 10))

    # Medium Performers
    medium_performers = sector_classifications.get('medium_performers', pd.DataFrame())
    if len(medium_performers) > 0:
        story.append(Paragraph(f"Medium Performers (25-50% overall completion): {len(medium_performers)} sectors",
                               heading2_style))
        story.append(Paragraph("These sectors completed a moderate percentage of fields:", normal_style))
        for _, row in medium_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg completion, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend (2023-2024): {row.get('trend_overall_23_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No medium performing sectors found", normal_style))

    story.append(Spacer(1, 10))

    # Low Performers
    low_performers = sector_classifications.get('low_performers', pd.DataFrame())
    if len(low_performers) > 0:
        story.append(
            Paragraph(f"Low Performers (<25% overall completion): {len(low_performers)} sectors", heading2_style))
        story.append(Paragraph("These sectors completed the lowest percentage of fields:", normal_style))
        for _, row in low_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg completion, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend (2023-2024): {row.get('trend_overall_23_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No low performing sectors found", normal_style))

    story.append(PageBreak())

    # Section 5: Performance Summary Table
    story.append(Paragraph("5. Performance Summary Table (2024)", heading1_style))
    story.append(Paragraph("All sectors sorted by overall completion percentage in 2024 (descending)", normal_style))

    # Add column explanations
    story.append(Paragraph("Column Explanations:", heading2_style))
    story.append(Paragraph("• Sector: ATECO sector letter", bullet_style))
    story.append(Paragraph("• Total Cos: Total number of companies in the sector", bullet_style))
    story.append(Paragraph("• 2024: Overall %: Percentage of all possible fields completed in 2024", bullet_style))
    story.append(Paragraph("• 2024: Avg Fields: Average number of fields completed per company in 2024", bullet_style))
    story.append(
        Paragraph("• 2024: % Any: Percentage of companies with at least 1 completed field in 2024", bullet_style))
    story.append(Paragraph("• Trend 23-24: Change in overall completion from 2023 to 2024 (positive = improvement)",
                           bullet_style))
    story.append(Spacer(1, 10))

    # Create summary table
    if len(summary_table_data) > 0:
        summary_data_table = [summary_table_headers] + summary_table_data

        summary_table = Table(summary_data_table, colWidths=[80, 90, 100, 100, 100, 90])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ]))

        story.append(summary_table)
    else:
        story.append(Paragraph("No summary data available", normal_style))

    story.append(Spacer(1, 20))

    # Section 6: Overall Engagement Trends
    story.append(Paragraph("6. Overall Engagement Trends (All Sectors Combined)", heading1_style))
    story.append(Paragraph("This table shows how overall completion has changed from 2023 to 2024:", normal_style))

    overall_data = [
        ['Year', 'Overall Completion %', 'Companies with Any Data %', 'Avg Fields/Company', 'Trend'],
        ['2023', f"{advanced_stats['yearly_stats']['2023']['overall_completion']:.1f}%",
         f"{advanced_stats['yearly_stats']['2023']['pct_with_any']:.1f}%",
         f"{advanced_stats['yearly_stats']['2023']['avg_fields_per_company']:.2f}", '-'],
        ['2024', f"{advanced_stats['yearly_stats']['2024']['overall_completion']:.1f}%",
         f"{advanced_stats['yearly_stats']['2024']['pct_with_any']:.1f}%",
         f"{advanced_stats['yearly_stats']['2024']['avg_fields_per_company']:.2f}",
         f"{advanced_stats['yearly_stats']['2024']['overall_completion'] - advanced_stats['yearly_stats']['2023']['overall_completion']:+.1f}%"],
        ['Overall Trend (2023-2024)', f"{advanced_stats['trend_stats']['overall_completion_growth']:+.1f}%",
         f"{advanced_stats['trend_stats']['pct_with_any_growth']:+.1f}%",
         f"{advanced_stats['trend_stats']['avg_fields_growth']:+.2f}",
         f"{advanced_stats['trend_stats']['overall_completion_growth']:+.1f}%"]
    ]

    overall_table = Table(overall_data, colWidths=[120, 140, 150, 140, 100])
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, 2), colors.HexColor('#f8f9fa')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('FONTNAME', (0, 3), (-1, 3), 'Helvetica-Bold'),
    ]))

    story.append(overall_table)
    story.append(Spacer(1, 15))

    # Add simple explanation of trends
    story.append(Paragraph("Trend Interpretation:", heading2_style))
    if trend_stats['overall_completion_growth'] > 0:
        story.append(Paragraph(
            f"✓ Overall completion increased by {trend_stats['overall_completion_growth']:.1f}% from 2023 to 2024",
            bullet_style))
    else:
        story.append(Paragraph(
            f"✗ Overall completion decreased by {abs(trend_stats['overall_completion_growth']):.1f}% from 2023 to 2024",
            bullet_style))

    if trend_stats['pct_with_any_growth'] > 0:
        story.append(
            Paragraph(f"✓ More companies now have at least some data (+{trend_stats['pct_with_any_growth']:.1f}%)",
                      bullet_style))
    else:
        story.append(Paragraph(f"✗ Fewer companies now have any data (-{abs(trend_stats['pct_with_any_growth']):.1f}%)",
                               bullet_style))

    # NEW SECTION: Company-Level Analysis
    story.append(PageBreak())
    story.append(Paragraph("7. Company-Level Performance Analysis", heading1_style))
    story.append(Paragraph("This section highlights individual company performance and improvement.", normal_style))
    story.append(Paragraph("Filter Criteria:", heading2_style))
    story.append(
        Paragraph("• Top performers: Companies with at least 4 out of 8 fields completed in 2024", bullet_style))
    story.append(
        Paragraph("• Most improved: Companies that improved by at least 2 fields from 2023 to 2024", bullet_style))

    if company_analysis is not None:
        # Add company analysis summary - FIXED: Adjusted column widths and font size
        summary_stats = company_analysis['summary_stats']

        story.append(Paragraph("Company-Level Summary Statistics:", heading2_style))

        company_summary_data = [
            ['Metric', 'Value'],
            ['Total Companies Analyzed', f"{summary_stats['total_companies_analyzed']:,}"],
            ['Companies with Data in 2024', f"{summary_stats['companies_with_data_2024']:,}"],
            ['Companies with ≥4/8 Fields in 2024', f"{summary_stats['companies_min_4_2024']:,}"],
            ['Companies with Significant Improvement (≥2 fields)', f"{summary_stats['companies_improved_min_2']:,}"],
            ['Average 2024 Score', f"{summary_stats['avg_2024_score']:.2f}"],
            ['Companies with Perfect Score in 2024 (8/8)', f"{summary_stats['companies_perfect_2024']:,}"],
            ['Companies Meeting High Performer Criteria', f"{summary_stats['companies_high_performers']:,}"],
            ['Companies Meeting Significant Improver Criteria', f"{summary_stats['companies_significant_improvers']:,}"]
        ]

        company_summary_table = Table(company_summary_data, colWidths=[250,180])
        company_summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 4),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
            ('WORDWRAP', (0, 0), (-1, -1), True),  # Added word wrap
        ]))
        story.append(company_summary_table)
        story.append(Spacer(1, 15))

        # FIXED: Add PageBreak before Top Performing Companies section
        story.append(PageBreak())

        # Top Performers in 2024 (with minimum 4/8) - FIXED: Now on new page
        story.append(Paragraph("Top Performing Companies in 2024 (Minimum 4/8 Fields):", heading2_style))
        story.append(Paragraph("These companies completed at least 4 fields in 2024, sorted by total fields completed:",
                               normal_style))

        top_performers_data = [['Rank', 'Company Name', 'Sector', '2023 Score', '2024 Score', 'Total Fields (2024)']]

        for i, (_, row) in enumerate(company_analysis['top_performers_2024'].iterrows(), 1):
            # Color coding based on score
            score_2024 = row['total_sum_2024']
            if score_2024 == 8:
                bg_color = colors.HexColor('#d4edda')  # Green for perfect score
            elif score_2024 >= 6:
                bg_color = colors.HexColor('#c3e6cb')  # Light green for high score
            elif score_2024 >= 4:
                bg_color = colors.HexColor('#f8f9fa')  # White for minimum threshold

            top_performers_data.append([
                f"{i}",
                str(row['Name'])[:40],  # Reduced from 50 to 40 characters
                str(row['ateco_letter']),
                f"{row['total_sum_2023']}",
                f"{row['total_sum_2024']}",
                f"{row['total_sum_2024']}/8"
            ])

        # FIXED: Adjusted column widths to prevent overlap
        top_performers_table = Table(top_performers_data, colWidths=[35, 160, 45, 65, 65, 80])
        top_performers_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),  # Reduced from 8 to 7
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 0), (-1, 0), 3),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 7),  # Reduced from 8 to 7
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 1), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
            ('WORDWRAP', (1, 1), (1, -1), True),  # Word wrap for company names
        ]))

        # Apply color coding to rows based on 2024 score
        for i in range(1, len(top_performers_data)):
            score_2024 = int(top_performers_data[i][4])  # Get 2024 score from table
            if score_2024 == 8:
                top_performers_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#d4edda')),
                    ('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'),
                ]))
            elif score_2024 >= 6:
                top_performers_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#c3e6cb')),
                ]))

        story.append(top_performers_table)
        story.append(Spacer(1, 15))

        # Top Improvers (minimum improvement of 2 fields)
        story.append(PageBreak())
        story.append(Paragraph("Most Improved Companies (Minimum Improvement of 2 Fields):", heading2_style))
        story.append(Paragraph("These companies improved by at least 2 fields from 2023 to 2024:", normal_style))

        top_improvers_data = [['Rank', 'Company Name', 'Sector', '2023 Score', '2024 Score', 'Improvement']]

        for i, (_, row) in enumerate(company_analysis['top_improvers'].iterrows(), 1):
            # Color coding based on improvement magnitude
            improvement = row['improvement']
            if improvement >= 6:
                bg_color = colors.HexColor('#d4edda')  # Green for major improvement
            elif improvement >= 4:
                bg_color = colors.HexColor('#fff3cd')  # Yellow for moderate improvement
            else:  # improvement >= 2
                bg_color = colors.HexColor('#f8f9fa')  # White for minimum improvement

            top_improvers_data.append([
                f"{i}",
                str(row['Name'])[:40],  # Reduced from 50 to 40 characters
                str(row['ateco_letter']),
                f"{row['total_sum_2023']}",
                f"{row['total_sum_2024']}",
                f"+{improvement}" if improvement > 0 else f"{improvement}"
            ])

        # FIXED: Adjusted column widths to prevent overlap
        top_improvers_table = Table(top_improvers_data, colWidths=[35, 160, 45, 65, 65, 70])
        top_improvers_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),  # Reduced from 8 to 7
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 0), (-1, 0), 3),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 7),  # Reduced from 8 to 7
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 1), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
            ('WORDWRAP', (1, 1), (1, -1), True),  # Word wrap for company names
        ]))

        # Apply color coding to rows based on improvement magnitude
        for i in range(1, len(top_improvers_data)):
            improvement_str = top_improvers_data[i][5]
            if improvement_str.startswith('+'):
                improvement_val = int(improvement_str[1:])  # Remove '+' and convert to int
                if improvement_val >= 6:
                    top_improvers_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#d4edda')),
                        ('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'),
                    ]))
                elif improvement_val >= 4:
                    top_improvers_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fff3cd')),
                        ('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'),
                    ]))

        story.append(top_improvers_table)

        # FIXED: Add PageBreak before Analysis Notes
        story.append(PageBreak())

        # Add notes about the analysis - FIXED: Now on new page
        story.append(Paragraph("Analysis Notes:", heading2_style))
        story.append(Paragraph("• Improvement is calculated as: 2024 Score - 2023 Score", bullet_style))
        story.append(Paragraph("• Only companies with improvement ≥ 2 fields are shown", bullet_style))
        story.append(Paragraph("• Only companies with 2024 score ≥ 4 are shown as top performers", bullet_style))
        story.append(Paragraph("• Perfect score = 8 out of 8 fields completed", bullet_style))
        story.append(Paragraph("• Major improvement = improvement of 6+ fields", bullet_style))
        story.append(Paragraph("• Moderate improvement = improvement of 4-5 fields", bullet_style))
        story.append(Paragraph("• Minimum improvement = improvement of 2-3 fields", bullet_style))

    else:
        story.append(Paragraph("Company-level analysis could not be performed due to missing data.", normal_style))
        story.append(
            Paragraph("Please ensure the dataset contains 'Name', 'total_sum_2023', and 'total_sum_2024' columns.",
                      bullet_style))

    story.append(PageBreak())

    # Section 8: Key Findings and Simple Insights
    story.append(Paragraph("8. Key Findings and Simple Insights", heading1_style))

    insights = [
        f"1. Overall Performance: In 2024, companies completed {key_stats['overall_completion']:.1f}% of all possible fields.",
        f"2. Company Participation: {key_stats['pct_with_any']:.1f}% of companies completed at least 1 field in 2024.",
        f"3. Full Compliance: Only {key_stats['pct_with_full']:.1f}% of companies completed all 8 fields in 2024.",
        f"4. Average Completion: Each company completed an average of {key_stats['avg_fields_per_company']:.1f} fields in 2024.",
        f"5. Most Common: The most common number of completed fields is {mode_str}.",
        f"6. Yearly Growth: From 2023 to 2024, completion increased by {trend_stats['overall_completion_growth']:+.1f}%.",
        f"7. Best Performing Sectors: {len(sector_classifications.get('high_performers', pd.DataFrame()))} sectors consistently completed >50% of fields.",
        f"8. Areas for Improvement: {len(sector_classifications.get('low_performers', pd.DataFrame()))} sectors completed <25% of fields."
    ]

    for insight in insights:
        story.append(Paragraph(insight, bullet_style))

    # Footer
    story.append(Spacer(1, 40))
    story.append(Paragraph(f"Report generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
                           ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9,
                                          alignment=TA_CENTER, textColor=colors.gray)))

    # Build the PDF
    try:
        doc.build(story)
        return pdf_filename
    except Exception as e:
        print(f"    ✗ Error building PDF: {e}")
        return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

print(f"\n✓ Analysis complete!")

# Generate PDF report
print(f"✓ Generating enhanced PDF report...")
pdf_file = create_pdf_report()

if pdf_file:
    print(f"✓ Enhanced PDF report successfully created: {pdf_file}")
else:
    print("✗ Failed to create PDF report")

print(f"\n✓ Image files created:")
valid_abs = [p for p in image_paths_absolute if p is not None]
valid_pct = [p for p in image_paths_percentage if p is not None]
print(f"  - Absolute charts: {len(valid_abs)} created")
print(f"  - Percentage charts: {len(valid_pct)} created")

print(f"\n✓ Dataset Specifications:")
print(f"  - Number of fields: 8")
print(f"  - Years analyzed: 2023, 2024")
print(f"  - Number of companies: {len(df)}")
print(f"  - Number of ATECO sectors: {len(sector_order)}")
print(
    f"  - Field names: Interviste, Survey_Online, Conference_Call, Visite_In_Loco, Seminari, Conferenze, Rapporti_Con_Agricoltori, Roadshow")

if company_analysis is not None:
    print(f"\n✓ Company Analysis Results:")
    print(f"  - Top performers in 2024 (≥4/8): {len(company_analysis['top_performers_2024'])} companies")
    print(f"  - Top improvers 2023-2024 (≥2 fields): {len(company_analysis['top_improvers'])} companies")
    print(f"  - Companies meeting high performer criteria: {company_analysis['summary_stats']['companies_min_4_2024']}")
    print(
        f"  - Companies meeting significant improver criteria: {company_analysis['summary_stats']['companies_improved_min_2']}")
else:
    print(f"\n⚠ Company analysis could not be performed. Check if 'Name' column exists in dataset.")