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


def reshaping_dataset(input_csv_path, output_csv_path, output_xlsx_path):
    """
    Process the CSV file by removing 2022 columns and renaming fields,
    then save as both CSV and XLSX files.

    Args:
        input_csv_path (str): Path to the input CSV file
        output_csv_path (str): Path to save the output CSV file
        output_xlsx_path (str): Path to save the output XLSX file
    """

    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Identify and remove columns for year 2022
    columns_to_remove = [col for col in df.columns if '_2022' in col]
    df = df.drop(columns=columns_to_remove)

    # Create a mapping dictionary for renaming columns
    rename_dict = {}

    # Rename Field1, Field2, Field3 for 2023 and 2024
    for year in ['_2023', '_2024']:
        if f'Field1{year}' in df.columns:
            rename_dict[f'Field1{year}'] = f'AB{year}'
        if f'Field2{year}' in df.columns:
            rename_dict[f'Field2{year}'] = f'AC{year}'
        if f'Field3{year}' in df.columns:
            rename_dict[f'Field3{year}'] = f'AD{year}'

    # Apply the renaming
    df = df.rename(columns=rename_dict)

    # Save to CSV
    df.to_csv(output_csv_path, index=False)

    # Save to XLSX
    df.to_excel(output_xlsx_path, index=False)

    print(f"Processed data saved to:")
    print(f"  CSV: {output_csv_path}")
    print(f"  XLSX: {output_xlsx_path}")

    # Return the dataframe for further inspection if needed
    return df


###############################################

"""
# OLD CODE
df = reshaping_dataset('Starting_Dataset.csv.csv', 'Starting_Dataset.csv', 'Tidier_Dataset.xlsx')
"""

###############################################


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

# Load the data
df = pd.read_csv('Tidier_Dataset.csv')

# Extract ATECO letter from the 'ateco' column
df['ateco_letter'] = df['ateco'].str[0]

# Define field groups - ONE GROUP with 3 fields and 2 years (2023, 2024)
field_groups = {
    'AB & AC & AD': {
        'fields': ['AB', 'AC', 'AD'],  # Three fields
        'color_base': '#3498db',  # Blue color
        'name': 'AB & AC & AD Fields',
        'max_fields': 3,  # Maximum of 3 fields per company
        'years': ['2023', '2024']  # Only 2 years
    }
}

# Calculate sum of ones for each group and year
for group_name, group_info in field_groups.items():
    for year in group_info['years']:
        field_columns = [f'{field}_{year}' for field in group_info['fields']]
        df[f'{group_name}_sum_{year}'] = df[field_columns].sum(axis=1)

# Calculate overall sum for each year
for year in field_groups['AB & AC & AD']['years']:
    all_field_columns = [f'{field}_{year}' for field in field_groups['AB & AC & AD']['fields']]
    df[f'total_sum_{year}'] = df[all_field_columns].sum(axis=1)


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
        light_values = np.linspace(0.85, 0.25, num_shades)
    else:
        light_values = np.linspace(0.25, 0.85, num_shades)

    for lightness in light_values:
        r, g, b = colorsys.hls_to_rgb(h, lightness, s)
        hex_color = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
        shades.append(hex_color)

    return shades


# Generate color gradients for the group (for values 0-3)
group_colors = {}
for group_name, group_info in field_groups.items():
    # Generate 4 colors for values 0, 1, 2, 3
    num_shades = group_info['max_fields'] + 1  # Including 0
    group_colors[group_name] = get_color_gradient(group_info['color_base'], num_shades, reverse=True)

# =============================================================================
# PREPARE DATA FOR VISUALIZATION
# =============================================================================

# Create detailed breakdown by ATECO letter and sum values for each group
detailed_breakdown = pd.DataFrame()

# Only process 2023 and 2024
for year in ['2023', '2024']:
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
# ENHANCED STATISTICAL ANALYSIS
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

        # Year-wise metrics (only 2023, 2024)
        for year in ['2023', '2024']:
            # Overall completion metrics
            total_fields_completed = sector_data[f'total_sum_{year}'].sum()
            max_possible = 3 * total_companies  # 3 fields per year
            overall_completion = (total_fields_completed / max_possible * 100) if max_possible > 0 else 0

            sector_metrics[f'overall_completion_{year}'] = overall_completion
            sector_metrics[
                f'avg_fields_per_company_{year}'] = total_fields_completed / total_companies if total_companies > 0 else 0

            # Percentage with any completion (at least 1 field)
            companies_with_any = len(sector_data[sector_data[f'total_sum_{year}'] > 0])
            sector_metrics[f'pct_with_any_{year}'] = (
                    companies_with_any / total_companies * 100) if total_companies > 0 else 0

            # Percentage with full completion (3 fields)
            companies_with_full = len(sector_data[sector_data[f'total_sum_{year}'] == 3])
            sector_metrics[f'pct_with_full_{year}'] = (
                    companies_with_full / total_companies * 100) if total_companies > 0 else 0

            # Group-specific metrics
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

        # Calculate trends (2023 to 2024 only)
        sector_metrics['trend_overall_23_24'] = sector_metrics.get('overall_completion_2024', 0) - sector_metrics.get(
            'overall_completion_2023', 0)
        sector_metrics['trend_any_23_24'] = sector_metrics.get('pct_with_any_2024', 0) - sector_metrics.get(
            'pct_with_any_2023', 0)

        # Calculate averages across years
        for metric in ['overall_completion', 'pct_with_any', 'pct_with_full']:
            values = [sector_metrics.get(f'{metric}_{year}', 0) for year in ['2023', '2024']]
            sector_metrics[f'avg_{metric}'] = np.mean(values)

        for group_name in field_groups.keys():
            completion_values = [sector_metrics.get(f'{group_name}_completion_{year}', 0) for year in ['2023', '2024']]
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
# ADVANCED STATISTICAL ANALYSIS
# =============================================================================

def calculate_advanced_statistics():
    """Calculate advanced statistical measures."""

    stats_dict = {}

    # Overall statistics for each year
    for year in ['2023', '2024']:
        year_stats = {
            'total_companies': len(df),
            'companies_with_any': len(df[df[f'total_sum_{year}'] > 0]),
            'companies_with_full': len(df[df[f'total_sum_{year}'] == 3]),  # Full = 3 fields
            'total_fields_completed': df[f'total_sum_{year}'].sum(),
            'max_possible_fields': len(df) * 3  # 3 fields per year
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
        for count in range(0, 4):  # 0 to 3 fields
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

    # Overall performance classification
    if 'avg_overall_completion' in performance_df.columns:
        overall_avg = performance_df['avg_overall_completion']
        classifications['high_performers'] = performance_df[overall_avg > 40] if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['medium_performers'] = performance_df[(overall_avg >= 20) & (overall_avg <= 40)] if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['low_performers'] = performance_df[overall_avg < 20] if len(
            performance_df) > 0 else pd.DataFrame()

    # Engagement level classification
    if 'avg_pct_with_any' in performance_df.columns:
        engagement_avg = performance_df['avg_pct_with_any']
        classifications['highly_engaged'] = performance_df[engagement_avg > 60] if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['moderately_engaged'] = performance_df[(engagement_avg >= 30) & (engagement_avg <= 60)] if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['low_engaged'] = performance_df[engagement_avg < 30] if len(
            performance_df) > 0 else pd.DataFrame()

    # Trend classification
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
# COMPANY-LEVEL ANALYSIS
# =============================================================================

def get_top_performing_companies(year='2024', top_n=20):
    """Get top performing companies based on total sum for a given year."""
    if 'Name' not in df.columns:
        print("Warning: 'Name' column not found in dataframe. Cannot identify companies.")
        return pd.DataFrame()

    # Sort by total_sum for the specified year (descending)
    top_companies = df.copy()

    # Check if the total_sum column exists
    total_sum_col = f'total_sum_{year}'
    if total_sum_col not in top_companies.columns:
        print(f"Warning: {total_sum_col} column not found in dataframe.")
        return pd.DataFrame()

    # Sort and get top N
    top_companies = top_companies.sort_values(by=total_sum_col, ascending=False).head(top_n)

    # Select relevant columns
    result_columns = ['Name', total_sum_col]

    # Add individual field information if available
    for field in ['AB', 'AC', 'AD']:
        field_col = f'{field}_{year}'
        if field_col in top_companies.columns:
            result_columns.append(field_col)

    # Add ATECO sector if available
    if 'ateco_letter' in top_companies.columns:
        result_columns.append('ateco_letter')

    return top_companies[result_columns]


def get_best_improvers(top_n=20):
    """Get companies with highest improvement from 2023 to 2024."""
    if 'Name' not in df.columns:
        print("Warning: 'Name' column not found in dataframe. Cannot identify companies.")
        return pd.DataFrame()

    # Check if both year columns exist
    if 'total_sum_2023' not in df.columns or 'total_sum_2024' not in df.columns:
        print("Warning: Required total_sum columns for 2023 and/or 2024 not found.")
        return pd.DataFrame()

    # Create a copy to avoid modifying original dataframe
    improvers_df = df.copy()

    # Calculate improvement (2024 - 2023)
    improvers_df['improvement'] = improvers_df['total_sum_2024'] - improvers_df['total_sum_2023']

    # Filter out companies that had 0 in both years (no change)
    # Or you might want to include all companies
    improvers_df = improvers_df[improvers_df['improvement'].notna()]

    # Sort by improvement (descending)
    improvers_df = improvers_df.sort_values(by='improvement', ascending=False).head(top_n)

    # Select relevant columns
    result_columns = ['Name', 'total_sum_2023', 'total_sum_2024', 'improvement']

    # Add ATECO sector if available
    if 'ateco_letter' in improvers_df.columns:
        result_columns.append('ateco_letter')

    return improvers_df[result_columns]


# =============================================================================
# CREATE VISUALIZATIONS - ABSOLUTE VALUES
# =============================================================================

def create_absolute_bar_chart(year_data, year, sector_order):
    """Create a bar chart with absolute counts."""
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
                for value in range(0, max_fields + 1):  # 0, 1, 2, 3
                    count = sector_row.iloc[0][f'{group_name}_{value}']
                    group_dict[value] = count
                sector_dict[group_name] = group_dict
            sector_data[sector] = sector_dict

    # Create the stacked bars
    group_order = ['AB & AC & AD']

    # Get colors for this group (4 colors for values 0-3, but we'll only use 1-3)
    colors_for_group = group_colors['AB & AC & AD']

    # Initialize bottoms for each bar
    bottoms = np.zeros(len(sector_order))

    # Stack from 1 to 3 (bottom to top: 1, 2, 3)
    for value in [1, 2, 3]:  # ONLY values >= 1
        counts = []
        for i, sector in enumerate(sector_order):
            if sector in sector_data:
                counts.append(sector_data[sector]['AB & AC & AD'].get(value, 0))
            else:
                counts.append(0)

        # Get color for this value (index 0=value0, 1=value1, etc.)
        color_idx = value  # 1, 2, 3
        color = colors_for_group[color_idx] if color_idx < len(colors_for_group) else colors_for_group[-1]

        bars = ax.bar(x_positions, counts, bottom=bottoms,
                      width=bar_width, color=color, alpha=0.9,
                      edgecolor='black', linewidth=0.5)
        bottoms += counts

    # Calculate total heights for sorting (sum of values 1, 2, 3)
    total_adopted = np.array(bottoms)  # This now only includes 1+2+3

    # Sort sectors by total adopted (descending)
    sorted_indices = np.argsort(total_adopted)[::-1]
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

    # Stack from 1 to 3 (bottom to top) for sorted data
    for value in [1, 2, 3]:  # ONLY values >= 1
        counts = []
        for sector in sorted_sectors:
            if sector in sorted_sector_data:
                counts.append(sorted_sector_data[sector]['AB & AC & AD'].get(value, 0))
            else:
                counts.append(0)

        color_idx = value
        color = colors_for_group[color_idx] if color_idx < len(colors_for_group) else colors_for_group[-1]

        ax.bar(x_positions_sorted, counts, bottom=bottoms_sorted,
               width=bar_width, color=color, alpha=0.9,
               edgecolor='black', linewidth=0.5)
        bottoms_sorted += counts

    # Customize the plot
    ax.set_title(f'{year} - AB & AC & AD Fields Completion',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ATECO Sector', fontsize=12)
    ax.set_ylabel('Number of Companies', fontsize=12)

    ax.set_xticks(x_positions_sorted)
    ax.set_xticklabels(sorted_sectors, rotation=45, ha='right', fontsize=12)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total count labels on top of bars
    max_height = max(bottoms_sorted) if len(bottoms_sorted) > 0 else 1
    for i, total in enumerate(bottoms_sorted):
        if total > 0:
            ax.text(i, total + max_height * 0.01, f'{int(total)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add 'n' labels at the bottom (total companies in sector)
    y_min, y_max = ax.get_ylim()

    if max_height > 0:
        offset = max_height * 0.08  # Reduced from 0.15 to 0.08
    else:
        offset = 0.3  # Reduced from 0.5 to 0.3

    ax.set_ylim(bottom=y_min - offset)

    n_label_y_position = y_min - offset * 0.15  # Reduced from 0.7 to 0.15 (MUCH CLOSER)

    for i, sector in enumerate(sorted_sectors):
        total_companies = sorted_sector_totals.get(sector, 0)
        ax.text(i, n_label_y_position, f'n={total_companies}',
                ha='center', va='top', fontsize=9, color='gray', fontweight='bold')

    # Create a comprehensive legend (ONLY for values 1, 2, 3)
    legend_items = []

    # Add group header
    legend_items.append((Line2D([0], [0], color='white'), f"AB & AC & AD Fields Completed:"))

    # Add items for each value in order 1, 2, 3 (as stacked)
    for value in [1, 2, 3]:
        color_idx = value
        color = colors_for_group[color_idx] if color_idx < len(colors_for_group) else colors_for_group[-1]

        if value == 1:
            label = f"  1 field"
        elif value == 2:
            label = f"  2 fields"
        else:
            label = f"  3 fields"

        legend_items.append((Patch(facecolor=color, edgecolor='black',
                                   alpha=0.9, linewidth=0.5),
                             label))

    # Create legend
    if legend_items:
        legend_handles, legend_labels = zip(*legend_items)

        ax.legend(legend_handles, legend_labels,
                  title='Fields Completed',
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=10,
                  title_fontsize=11,
                  frameon=True,
                  fancybox=True,
                  shadow=True)

    return fig, sorted_sectors


# =============================================================================
# CREATE VISUALIZATIONS - PERCENTAGES
# =============================================================================

def create_percentage_bar_chart(year_data, year, sector_order):
    """Create a stacked bar chart showing percentage of total fields completed."""
    fig, ax = plt.subplots(figsize=(16, 8))

    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.18, top=0.92)

    # Calculate percentages for each sector, stratified by completion level (ONLY 1, 2, 3)
    sector_stratified_data = {}
    sector_totals = {}
    sector_total_percentages = {}

    for sector in sector_order:
        sector_row = year_data[year_data['ateco_letter'] == sector]
        if not sector_row.empty:
            total_companies = sector_row.iloc[0]['total_companies']
            sector_totals[sector] = total_companies

            # Calculate total possible fields (3 fields per company)
            total_possible_fields = 3 * total_companies if total_companies > 0 else 1

            # For each completion level (1, 2, 3), calculate contribution percentage
            stratified_dict = {}
            total_percentage = 0

            for value in range(1, 4):  # ONLY 1, 2, 3 (exclude 0)
                count = sector_row.iloc[0].get(f'AB & AC & AD_{value}', 0)
                # Each company with 'value' completed fields contributes:
                # (value * count) fields completed
                fields_completed = value * count
                # Percentage contribution to total possible fields
                percentage = (fields_completed / total_possible_fields * 100) if total_possible_fields > 0 else 0
                stratified_dict[value] = percentage
                total_percentage += percentage

            sector_stratified_data[sector] = stratified_dict
            sector_total_percentages[sector] = total_percentage

    # Sort sectors by total percentage (descending)
    sorted_sectors = sorted(sector_total_percentages.items(), key=lambda x: x[1], reverse=True)
    sorted_sector_names = [sector for sector, _ in sorted_sectors]

    # Get colors for this group
    colors_for_group = group_colors['AB & AC & AD']

    # Create the stacked bars - ONLY VALUES 1, 2, 3
    x_positions_sorted = np.arange(len(sorted_sector_names))
    bottoms = np.zeros(len(sorted_sector_names))

    # Stack from 1 to 3 (bottom to top)
    for value in [1, 2, 3]:  # ONLY values >= 1
        # Get percentages for this value across all sorted sectors
        percentages = []
        for sector in sorted_sector_names:
            if sector in sector_stratified_data:
                percentages.append(sector_stratified_data[sector].get(value, 0))
            else:
                percentages.append(0)

        if sum(percentages) > 0:  # Only add if there are any
            color_idx = value
            color = colors_for_group[color_idx] if color_idx < len(colors_for_group) else colors_for_group[-1]

            # Plot this segment
            ax.bar(x_positions_sorted, percentages, bottom=bottoms,
                   width=0.7, color=color, alpha=0.9,
                   edgecolor='black', linewidth=0.5)
            bottoms += percentages

    # Customize the plot
    ax.set_title(f'{year} - Percentage of Total AB & AC & AD Fields Completed',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ATECO Sector', fontsize=12)
    ax.set_ylabel('Percentage of Total Possible Fields Completed (%)', fontsize=12)

    ax.set_xticks(x_positions_sorted)
    ax.set_xticklabels(sorted_sector_names, rotation=45, ha='right', fontsize=12)

    # Set y-axis limit
    max_bar_height = max(bottoms) if len(bottoms) > 0 else 0
    # Add 10% padding, but don't exceed 100%
    y_max = min(100, max_bar_height * 1.10)
    if y_max < 10:
        y_max = 10

    y_min = 0
    offset = y_max * 0.05  # Reduced from 0.10 to 0.05

    ax.set_ylim(bottom=y_min - offset, top=y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total percentage labels on top of bars
    for i, (sector, total_pct) in enumerate(zip(sorted_sector_names, bottoms)):
        if total_pct > 0:
            ax.text(i, total_pct + y_max * 0.01, f'{total_pct:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add 'n' labels at the bottom
    n_label_y_position = y_min - offset * 0.15  # Reduced from 0.5 to 0.15 (MUCH CLOSER)

    for i, sector in enumerate(sorted_sector_names):
        total_companies = sector_totals.get(sector, 0)
        ax.text(i, n_label_y_position, f'n={total_companies}',
                ha='center', va='top', fontsize=9, color='gray', fontweight='bold')

    # Create a comprehensive legend (ONLY for values 1, 2, 3)
    legend_items = []

    # Add group header - FIXED: Added the missing label
    legend_items.append((Line2D([0], [0], color='white'), f"AB & AC & AD Fields Completed:"))

    # Add items for each value in order 1, 2, 3 (as stacked)
    for value in [1, 2, 3]:
        color_idx = value
        color = colors_for_group[color_idx] if color_idx < len(colors_for_group) else colors_for_group[-1]

        if value == 1:
            label = f"  1 field"
        elif value == 2:
            label = f"  2 fields"
        else:
            label = f"  3 fields"

        legend_items.append((Patch(facecolor=color, edgecolor='black',
                                   alpha=0.9, linewidth=0.5),
                             label))

    # Create legend
    if legend_items:
        legend_handles, legend_labels = zip(*legend_items)

        ax.legend(legend_handles, legend_labels,
                  title='Fields Completed',
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=10,
                  title_fontsize=11,
                  frameon=True,
                  fancybox=True,
                  shadow=True)

    return fig, sorted_sector_names


# =============================================================================
# CREATE ALL VISUALIZATIONS
# =============================================================================

# Create visualizations for each year
image_paths_absolute = []
image_paths_percentage = []
all_sorted_sectors = {}

print("  - Creating absolute value charts...")
for year in ['2023', '2024']:
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
        import traceback

        print(f"    Detailed error: {traceback.format_exc()[:200]}...")
        image_paths_absolute.append(None)

print("  - Creating percentage charts...")
for year in ['2023', '2024']:
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

# Get company-level analysis
print("  - Analyzing company performance...")
top_performers_2024 = get_top_performing_companies(year='2024', top_n=20)
best_improvers = get_best_improvers(top_n=20)

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
# ENHANCED PDF REPORT WITH STATISTICAL ANALYSIS
# =============================================================================

def create_pdf_report():
    # Import colors locally to avoid conflicts
    from reportlab.lib import colors

    # Create PDF document
    pdf_filename = f"AB & AC & AD_Fields_Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
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

    # Create story (content) list
    story = []

    # Title page
    story.append(Paragraph("AB, AC, AD Fields Analysis by ATECO Sector", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Analysis Date: {datetime.datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph("Dataset: Starting_Dataset.csv (AB & AC & AD Fields for 2023-2024)", styles['Normal']))
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
        ['Companies with All 3 Fields (2024)', f"{key_stats['pct_with_full']:.1f}%"],
        ['Average Fields per Company (2024)', f"{key_stats['avg_fields_per_company']:.2f}"],
        ['Median Fields per Company (2024)', f"{key_stats['median_fields']:.1f}"],
        ['Standard Deviation (2024)', f"{key_stats['std_fields']:.2f}"],
        ['Trend 2023-2024', f"{trend_stats['overall_completion_growth']:+.1f}%"],
        ['Engagement Growth 2023-2024', f"{trend_stats['pct_with_any_growth']:+.1f}%"],
        ['High Performing Sectors (>40%)', f"{len(sector_classifications.get('high_performers', pd.DataFrame()))}"],
        ['Medium Performing Sectors (20-40%)',
         f"{len(sector_classifications.get('medium_performers', pd.DataFrame()))}"],
        ['Low Performing Sectors (<20%)', f"{len(sector_classifications.get('low_performers', pd.DataFrame()))}"]
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
    story.append(Paragraph("This section shows how many companies completed 0, 1, 2, or 3 fields.", normal_style))

    # Distribution statistics table
    dist_headers = ['Fields Completed', 'Companies (2023)', '%', 'Companies (2024)', '%']
    dist_data = [dist_headers]

    # Show distribution for 0, 1, 2, 3
    for fields in range(0, 4):
        row = [f"{fields} field{'s' if fields != 1 else ''}"]
        for year in ['2023', '2024']:
            year_dist = advanced_stats['distribution_data'][year]
            if fields < len(year_dist):
                companies = year_dist[fields]['companies']
                percentage = year_dist[fields]['percentage']
                row.append(f"{companies:,}")
                row.append(f"{percentage:.1f}%")
            else:
                row.append("0")
                row.append("0.0%")
        dist_data.append(row)

    dist_table = Table(dist_data, colWidths=[120, 100, 80, 100, 80])
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

    # Add statistical summary
    story.append(Paragraph("Key Points:", heading2_style))

    # Calculate mode
    mode_values = df['total_sum_2024'].mode()
    mode_str = f"{mode_values.iloc[0]}" if not mode_values.empty else 'N/A'

    story.append(Paragraph(f"• Most common: {mode_str} field{'s' if mode_str != '1' else ''} completed", bullet_style))
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
    story.append(Paragraph("2. Absolute Count Charts by Year", heading1_style))
    story.append(Paragraph("Each bar shows the number of companies with 0, 1, 2, or 3 completed fields in each sector.",
                           normal_style))
    story.append(Paragraph("How to read: Stacked colors show different completion levels (see legend)", bullet_style))
    story.append(Paragraph("Sectors sorted by number of companies with any completed fields", bullet_style))
    story.append(Paragraph("n= shows total number of companies in that sector", bullet_style))

    for year, image_path in zip(['2023', '2024'], image_paths_absolute):
        if image_path and os.path.exists(image_path):
            content_to_keep_together = []
            content_to_keep_together.append(
                Paragraph(f"{year} - AB & AC & AD Fields Completion (Absolute Counts)", heading2_style))
            try:
                content_to_keep_together.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                content_to_keep_together.append(Paragraph("Image not available", normal_style))
            content_to_keep_together.append(Spacer(1, 5))

            story.append(KeepTogether(content_to_keep_together))

    story.append(PageBreak())

    # Section 3: Percentage Visualizations
    story.append(Paragraph("3. Percentage Charts by Year", heading1_style))
    story.append(
        Paragraph("Each bar shows the percentage of total possible fields completed in each sector.", normal_style))
    story.append(Paragraph("How to read: Taller bars = higher percentage of fields completed", bullet_style))
    story.append(Paragraph("Note: 100% would mean ALL companies completed ALL 3 fields", bullet_style))
    story.append(Paragraph("Sectors sorted by percentage of fields completed (descending)", bullet_style))
    story.append(Paragraph("n= shows total number of companies in that sector", bullet_style))

    for year, image_path in zip(['2023', '2024'], image_paths_percentage):
        if image_path and os.path.exists(image_path):
            content_to_keep_together = []
            content_to_keep_together.append(
                Paragraph(f"{year} - AB & AC & AD Fields Completion Rate (%)", heading2_style))
            try:
                content_to_keep_together.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                content_to_keep_together.append(Paragraph("Image not available", normal_style))
            content_to_keep_together.append(Spacer(1, 5))

            story.append(KeepTogether(content_to_keep_together))

    story.append(PageBreak())

    # Section 4: Sector Performance Classification
    story.append(Paragraph("4. Sector Performance Classification", heading1_style))
    story.append(Paragraph("Note: Classification based on average completion across 2023-2024", normal_style))
    story.append(Spacer(1, 10))

    # High Performers
    high_performers = sector_classifications.get('high_performers', pd.DataFrame())
    if len(high_performers) > 0:
        story.append(Paragraph(f"High Performing Sectors (>40%): {len(high_performers)} sectors", heading2_style))
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
        story.append(Paragraph(f"Medium Performing Sectors (20-40%): {len(medium_performers)} sectors", heading2_style))
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
        story.append(Paragraph(f"Low Performing Sectors (<20%): {len(low_performers)} sectors", heading2_style))
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
    story.append(Paragraph("• 2024: Avg Fields: Average number of fields completed per company (0-3)", bullet_style))
    story.append(Paragraph("• 2024: % Any: Percentage of companies with at least 1 completed field", bullet_style))
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
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
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
        story.append(Paragraph(
            f"✓ More companies now have at least some data (+{trend_stats['pct_with_any_growth']:.1f}%)",
            bullet_style))
    else:
        story.append(Paragraph(
            f"✗ Fewer companies now have any data (-{abs(trend_stats['pct_with_any_growth']):.1f}%)",
            bullet_style))

    story.append(PageBreak())

    # NEW SECTION: Top Performing Companies
    story.append(Paragraph("7. Top Performing Companies in 2024", heading1_style))
    story.append(
        Paragraph("This section identifies the companies with the highest completion rates in 2024.", normal_style))
    story.append(Paragraph("Companies are ranked by total fields completed (maximum = 3 fields).", bullet_style))

    if not top_performers_2024.empty:
        # Prepare table data for top performers
        top_performers_data = []

        # Determine table structure based on available columns
        if 'ateco_letter' in top_performers_2024.columns:
            headers = ['Rank', 'Company Name', 'Sector', 'Total Fields (2024)', 'AB', 'AC', 'AD']
        else:
            headers = ['Rank', 'Company Name', 'Total Fields (2024)', 'AB', 'AC', 'AD']

        top_performers_data.append(headers)

        for idx, (_, row) in enumerate(top_performers_2024.iterrows(), 1):
            company_name = str(row['Name'])[:40]  # Truncate long names
            total_fields = int(row['total_sum_2024'])

            # Get individual field values
            ab_value = int(row.get('AB_2024', 0))
            ac_value = int(row.get('AC_2024', 0))
            ad_value = int(row.get('AD_2024', 0))

            if 'ateco_letter' in top_performers_2024.columns:
                sector = str(row['ateco_letter'])
                table_row = [str(idx), company_name, sector, str(total_fields),
                             str(ab_value), str(ac_value), str(ad_value)]
            else:
                table_row = [str(idx), company_name, str(total_fields),
                             str(ab_value), str(ac_value), str(ad_value)]

            top_performers_data.append(table_row)

        # Create table with appropriate column widths
        if 'ateco_letter' in top_performers_2024.columns:
            col_widths = [50, 200, 60, 90, 50, 50, 50]
        else:
            col_widths = [50, 250, 90, 50, 50, 50]

        top_performers_table = Table(top_performers_data, colWidths=col_widths)
        top_performers_table.setStyle(TableStyle([
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
            # Highlight companies with perfect score (3 fields)
            ('BACKGROUND', (len(headers) - 4 if 'ateco_letter' in headers else len(headers) - 3, 1),
             (len(headers) - 4 if 'ateco_letter' in headers else len(headers) - 3, -1),
             colors.HexColor('#d4edda')),  # Light green for total
        ]))

        # Additional highlighting for perfect scores
        for i in range(1, len(top_performers_data)):
            if top_performers_data[i][3 if 'ateco_letter' in headers else 2] == '3':
                # Highlight the entire row for companies with 3 fields
                top_performers_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#d4edda')),
                ]))

        story.append(top_performers_table)
        story.append(Spacer(1, 10))

        # Add analysis notes
        story.append(Paragraph("Analysis Notes:", heading2_style))

        # Count perfect scores
        perfect_scores = len(top_performers_2024[top_performers_2024['total_sum_2024'] == 3])
        story.append(
            Paragraph(f"• {perfect_scores} out of {len(top_performers_2024)} top companies completed all 3 fields",
                      bullet_style))

        # Calculate average among top performers
        avg_top_performers = top_performers_2024['total_sum_2024'].mean()
        story.append(
            Paragraph(f"• Average among top {len(top_performers_2024)} performers: {avg_top_performers:.2f} fields",
                      bullet_style))

        # Identify sectors with most top performers
        if 'ateco_letter' in top_performers_2024.columns:
            sector_counts = top_performers_2024['ateco_letter'].value_counts()
            if not sector_counts.empty:
                top_sector = sector_counts.index[0]
                top_sector_count = sector_counts.iloc[0]
                story.append(
                    Paragraph(f"• Sector with most top performers: {top_sector} ({top_sector_count} companies)",
                              bullet_style))
    else:
        story.append(
            Paragraph("Company performance data not available. Please check if 'Name' column exists in the dataset.",
                      normal_style))

    story.append(PageBreak())

    # NEW SECTION: Best Improvers
    story.append(Paragraph("8. Most Improved Companies (2023-2024)", heading1_style))
    story.append(Paragraph("This section identifies companies that showed the most improvement from 2023 to 2024.",
                           normal_style))
    story.append(Paragraph("Improvement is calculated as: (2024 total fields) - (2023 total fields)", bullet_style))

    if not best_improvers.empty:
        # Prepare table data for best improvers
        improvers_data = []

        # Determine table structure based on available columns
        if 'ateco_letter' in best_improvers.columns:
            headers = ['Rank', 'Company Name', 'Sector', '2023 Fields', '2024 Fields', 'Improvement']
        else:
            headers = ['Rank', 'Company Name', '2023 Fields', '2024 Fields', 'Improvement']

        improvers_data.append(headers)

        for idx, (_, row) in enumerate(best_improvers.iterrows(), 1):
            company_name = str(row['Name'])[:40]  # Truncate long names
            fields_2023 = int(row['total_sum_2023'])
            fields_2024 = int(row['total_sum_2024'])
            improvement = int(row['improvement'])

            if 'ateco_letter' in best_improvers.columns:
                sector = str(row['ateco_letter'])
                table_row = [str(idx), company_name, sector, str(fields_2023),
                             str(fields_2024), f"+{improvement}" if improvement > 0 else str(improvement)]
            else:
                table_row = [str(idx), company_name, str(fields_2023),
                             str(fields_2024), f"+{improvement}" if improvement > 0 else str(improvement)]

            improvers_data.append(table_row)

        # Create table with appropriate column widths
        if 'ateco_letter' in best_improvers.columns:
            col_widths = [50, 180, 60, 70, 70, 80]
        else:
            col_widths = [50, 230, 70, 70, 80]

        improvers_table = Table(improvers_data, colWidths=col_widths)
        improvers_table.setStyle(TableStyle([
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
            # Color code improvement column
            ('TEXTCOLOR', (len(headers) - 1, 1), (len(headers) - 1, -1), colors.green),
        ]))

        # Additional highlighting for different improvement levels
        for i in range(1, len(improvers_data)):
            improvement_str = improvers_data[i][-1]
            if improvement_str.startswith('+'):
                try:
                    improvement_val = int(improvement_str[1:])
                    if improvement_val == 3:  # Went from 0 to 3
                        improvers_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#d4edda')),  # Light green
                        ]))
                    elif improvement_val == 2:  # Went from 0/1 to 2/3
                        improvers_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fff3cd')),  # Light yellow
                        ]))
                except:
                    pass

        story.append(improvers_table)
        story.append(Spacer(1, 10))

        # Add analysis notes
        story.append(Paragraph("Analysis Notes:", heading2_style))

        # Calculate statistics
        total_improvement = best_improvers['improvement'].sum()
        avg_improvement = best_improvers['improvement'].mean()
        max_improvement = best_improvers['improvement'].max()
        companies_with_max_improvement = best_improvers[best_improvers['improvement'] == max_improvement]

        story.append(
            Paragraph(f"• Total improvement across top {len(best_improvers)} improvers: +{total_improvement} fields",
                      bullet_style))
        story.append(Paragraph(f"• Average improvement: +{avg_improvement:.2f} fields per company", bullet_style))
        story.append(Paragraph(f"• Maximum improvement: +{max_improvement} fields", bullet_style))

        # List companies with maximum improvement
        if len(companies_with_max_improvement) > 0:
            if len(companies_with_max_improvement) <= 3:
                names = ", ".join(companies_with_max_improvement['Name'].str[:30].tolist())
                story.append(
                    Paragraph(f"• Companies with maximum improvement (+{max_improvement}): {names}", bullet_style))
            else:
                story.append(Paragraph(
                    f"• {len(companies_with_max_improvement)} companies improved by +{max_improvement} fields",
                    bullet_style))

        # Identify sectors with most improvers
        if 'ateco_letter' in best_improvers.columns:
            sector_counts = best_improvers['ateco_letter'].value_counts()
            if not sector_counts.empty:
                top_sector = sector_counts.index[0]
                top_sector_count = sector_counts.iloc[0]
                story.append(Paragraph(f"• Sector with most improvers: {top_sector} ({top_sector_count} companies)",
                                       bullet_style))

        # Show examples of different improvement patterns
        if len(best_improvers) >= 3:
            story.append(Paragraph("• Improvement patterns:", bullet_style))

            # Find examples of different improvement levels
            for imp_level in [3, 2, 1]:
                examples = best_improvers[best_improvers['improvement'] == imp_level]
                if not examples.empty:
                    example_name = examples.iloc[0]['Name'][:30]
                    from_fields = int(examples.iloc[0]['total_sum_2023'])
                    to_fields = int(examples.iloc[0]['total_sum_2024'])
                    story.append(Paragraph(f"  - {example_name}: {from_fields} → {to_fields} fields (+{imp_level})",
                                           ParagraphStyle('BulletIndent', parent=bullet_style, leftIndent=40)))
    else:
        story.append(
            Paragraph("Company improvement data not available. Please check if 'Name' column exists in the dataset.",
                      normal_style))

    story.append(PageBreak())

    # Section 9: Key Findings and Insights (renumbered from original Section 7)
    story.append(Paragraph("9. Key Findings and Insights", heading1_style))

    insights = [
        f"1. Overall Performance: In 2024, companies completed {key_stats['overall_completion']:.1f}% of all possible AB & AC & AD fields.",
        f"2. Company Participation: {key_stats['pct_with_any']:.1f}% of companies completed at least 1 field in 2024.",
        f"3. Full Compliance: Only {key_stats['pct_with_full']:.1f}% of companies completed all 3 fields in 2024.",
        f"4. Average Completion: Each company completed an average of {key_stats['avg_fields_per_company']:.1f} fields in 2024.",
        f"5. Most Common: The most common number of completed fields is {mode_str}.",
        f"6. Yearly Growth: From 2023 to 2024, completion changed by {trend_stats['overall_completion_growth']:+.1f}%.",
        f"7. Best Performing Sectors: {len(sector_classifications.get('high_performers', pd.DataFrame()))} sectors consistently completed >40% of fields.",
        f"8. Areas for Improvement: {len(sector_classifications.get('low_performers', pd.DataFrame()))} sectors completed <20% of fields."
    ]

    for insight in insights:
        story.append(Paragraph(insight, bullet_style))

    # Add insights about top companies
    if not top_performers_2024.empty:
        perfect_count = len(top_performers_2024[top_performers_2024['total_sum_2024'] == 3])
        story.append(
            Paragraph(f"9. Top Performers: {perfect_count} companies achieved perfect scores (3/3 fields) in 2024.",
                      bullet_style))

    if not best_improvers.empty and not best_improvers.empty:
        max_imp = best_improvers['improvement'].max() if 'improvement' in best_improvers.columns else 0
        if max_imp > 0:
            story.append(
                Paragraph(f"10. Most Improved: The best improver increased by {max_imp} fields from 2023 to 2024.",
                          bullet_style))

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

print(f"\n✓ Company analysis:")
print(f"  - Top performers in 2024: {len(top_performers_2024)} companies identified")
print(f"  - Best improvers (2023-2024): {len(best_improvers)} companies identified")

# Display a sample of top performers
if not top_performers_2024.empty:
    print(f"\n✓ Sample of top 5 performers in 2024:")
    for i, (_, row) in enumerate(top_performers_2024.head(5).iterrows(), 1):
        name = str(row['Name'])[:50]
        total = int(row['total_sum_2024'])
        print(f"  {i}. {name}: {total} fields")

if not best_improvers.empty:
    print(f"\n✓ Sample of top 5 improvers (2023-2024):")
    for i, (_, row) in enumerate(best_improvers.head(5).iterrows(), 1):
        name = str(row['Name'])[:50]
        from_fields = int(row['total_sum_2023'])
        to_fields = int(row['total_sum_2024'])
        improvement = int(row['improvement'])
        print(f"  {i}. {name}: {from_fields} → {to_fields} fields (+{improvement})")

print(f"\n✓ Analysis completed successfully!")