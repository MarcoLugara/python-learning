import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import datetime
import os
import colorsys
from scipy import stats

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

# Calculate overall sum for each year
for year in ['2022', '2023', '2024']:
    all_field_columns = [f'Field{i}_{year}' for i in range(1, 17)]
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

    # Generate shades (skip 0, so start from 1)
    shades = []
    if reverse:
        light_values = np.linspace(0.7, 0.3, num_shades)  # Dark to light
    else:
        light_values = np.linspace(0.3, 0.7, num_shades)  # Light to dark

    for lightness in light_values:
        # Keep hue and saturation, adjust lightness
        r, g, b = colorsys.hls_to_rgb(h, lightness, s)
        # Convert back to hex
        hex_color = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
        shades.append(hex_color)

    return shades


# Generate color gradients for each group (excluding 0)
group_colors = {}
for group_name, group_info in field_groups.items():
    # Generate colors for each possible value (1 to max_fields, excluding 0)
    num_shades = group_info['max_fields']  # Excluding 0
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

        # Year-wise metrics
        for year in ['2022', '2023', '2024']:
            # Overall completion metrics
            total_fields_completed = sector_data[f'total_sum_{year}'].sum()
            max_possible = 16 * total_companies
            overall_completion = (total_fields_completed / max_possible * 100) if max_possible > 0 else 0

            sector_metrics[f'overall_completion_{year}'] = overall_completion
            sector_metrics[
                f'avg_fields_per_company_{year}'] = total_fields_completed / total_companies if total_companies > 0 else 0

            # Percentage with any completion
            companies_with_any = len(sector_data[sector_data[f'total_sum_{year}'] > 0])
            sector_metrics[f'pct_with_any_{year}'] = (
                        companies_with_any / total_companies * 100) if total_companies > 0 else 0

            # Percentage with full completion (16 fields)
            companies_with_full = len(sector_data[sector_data[f'total_sum_{year}'] == 16])
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

        # Calculate trends
        sector_metrics['trend_overall_22_24'] = sector_metrics.get('overall_completion_2024', 0) - sector_metrics.get(
            'overall_completion_2022', 0)
        sector_metrics['trend_any_22_24'] = sector_metrics.get('pct_with_any_2024', 0) - sector_metrics.get(
            'pct_with_any_2022', 0)

        # Calculate averages across years
        for metric in ['overall_completion', 'pct_with_any', 'pct_with_full']:
            values = [sector_metrics.get(f'{metric}_{year}', 0) for year in ['2022', '2023', '2024']]
            sector_metrics[f'avg_{metric}'] = np.mean(values)

        for group_name in field_groups.keys():
            completion_values = [sector_metrics.get(f'{group_name}_completion_{year}', 0) for year in
                                 ['2022', '2023', '2024']]
            any_values = [sector_metrics.get(f'{group_name}_pct_any_{year}', 0) for year in ['2022', '2023', '2024']]

            sector_metrics[f'avg_{group_name}_completion'] = np.mean(completion_values)
            sector_metrics[f'avg_{group_name}_pct_any'] = np.mean(any_values)

            # Group-specific trends
            sector_metrics[f'trend_{group_name}_completion_22_24'] = (
                    sector_metrics.get(f'{group_name}_completion_2024', 0) -
                    sector_metrics.get(f'{group_name}_completion_2022', 0)
            )

        metrics_list.append(sector_metrics)

    return pd.DataFrame(metrics_list)


# =============================================================================
# ADVANCED STATISTICAL ANALYSIS
# =============================================================================

def calculate_advanced_statistics():
    """Calculate advanced statistical measures."""

    stats_dict = {}

    # Overall statistics
    for year in ['2022', '2023', '2024']:
        year_stats = {
            'total_companies': len(df),
            'companies_with_any': len(df[df[f'total_sum_{year}'] > 0]),
            'companies_with_full': len(df[df[f'total_sum_{year}'] == 16]),
            'total_fields_completed': df[f'total_sum_{year}'].sum(),
            'max_possible_fields': len(df) * 16
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
    for year in ['2022', '2023', '2024']:
        distribution = []
        total_companies = len(df)
        for count in range(0, 17):  # 0 to 16 fields
            companies = len(df[df[f'total_sum_{year}'] == count])
            percentage = (companies / total_companies * 100) if total_companies > 0 else 0
            distribution.append({
                'fields': count,
                'companies': companies,
                'percentage': percentage
            })
        distribution_data[year] = distribution

    # Trend analysis
    trend_stats = {
        'overall_completion_growth': stats_dict['2024']['overall_completion'] - stats_dict['2022'][
            'overall_completion'],
        'pct_with_any_growth': stats_dict['2024']['pct_with_any'] - stats_dict['2022']['pct_with_any'],
        'avg_fields_growth': stats_dict['2024']['avg_fields_per_company'] - stats_dict['2022']['avg_fields_per_company']
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
    if 'trend_overall_22_24' in performance_df.columns:
        trend = performance_df['trend_overall_22_24']
        classifications['biggest_improvers'] = performance_df.nlargest(min(5, len(performance_df)),
                                                                       'trend_overall_22_24') if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['biggest_decliners'] = performance_df.nsmallest(min(5, len(performance_df)),
                                                                        'trend_overall_22_24') if len(
            performance_df) > 0 else pd.DataFrame()
        classifications['consistent_performers'] = performance_df[abs(trend) < 5] if len(
            performance_df) > 0 else pd.DataFrame()  # Less than 5% change

    # Group leaders
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
# CREATE VISUALIZATIONS - ABSOLUTE VALUES
# =============================================================================

def create_absolute_bar_chart(year_data, year, sector_order):
    """Create a bar chart with absolute counts (excluding zero values)."""
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.15, top=0.92)

    # Set up bar positions and width
    x_positions = np.arange(len(sector_order))
    bar_width = 0.7

    # Prepare data structure: sector -> group -> value -> count
    sector_data = {}
    for sector in sector_order:
        sector_row = year_data[year_data['ateco_letter'] == sector]
        if not sector_row.empty:
            sector_dict = {}
            for group_name in field_groups.keys():
                group_dict = {}
                max_fields = field_groups[group_name]['max_fields']
                for value in range(1, max_fields + 1):  # Start from 1, excluding 0
                    count = sector_row.iloc[0][f'{group_name}_{value}']
                    group_dict[value] = count
                sector_dict[group_name] = group_dict
            sector_data[sector] = sector_dict

    # Create the stacked bars
    group_order = ['Red_Group', 'Blue_Group', 'Green_Group']

    # Initialize bottoms for each bar
    bottoms = np.zeros(len(sector_order))

    # Keep track of which colors we've added to legend
    legend_added = {}

    # Stack in order: Red (1-8), Blue (1-5), Green (1-3)
    # For visual clarity, we'll stack from lowest to highest value within each group
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
                color_idx = value - 1  # 0 = 1 field, max_fields-1 = max_fields fields
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
    sorted_heights = total_heights[sorted_indices]

    # Reorder the bars
    for i in range(len(ax.patches)):
        ax.patches[i].remove()

    # Recreate bars in sorted order
    x_positions_sorted = np.arange(len(sorted_sectors))
    bottoms_sorted = np.zeros(len(sorted_sectors))

    # Create sorted sector data
    sorted_sector_data = {}
    for i, sector in enumerate(sorted_sectors):
        if sector in sector_data:
            sorted_sector_data[sector] = sector_data[sector]

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

    # Set x-ticks
    ax.set_xticks(x_positions_sorted)
    ax.set_xticklabels(sorted_sectors, rotation=45, ha='right', fontsize=11)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total count labels on top of bars
    for i, total in enumerate(bottoms_sorted):
        if total > 0:
            ax.text(i, total + max(bottoms_sorted) * 0.01, f'{int(total)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Create a comprehensive legend
    legend_items = []
    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Add group header
        legend_items.append((Line2D([0], [0], color='white'), f"<b>{group_info['name']}:</b>"))

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

        # Create legend outside the plot
        ax.legend(legend_handles, legend_labels,
                  title='Completed Fields by Group',
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=10,
                  title_fontsize=11,
                  frameon=True,
                  fancybox=True,
                  shadow=True)

    return fig, sorted_sectors


# =============================================================================
# CREATE VISUALIZATIONS - PERCENTAGES (FIXED)
# =============================================================================

def create_percentage_bar_chart(year_data, year, sector_order):
    """Create a bar chart with percentages that don't exceed 100%."""
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.15, top=0.92)

    # Set up bar positions and width
    x_positions = np.arange(len(sector_order))
    bar_width = 0.7

    # Prepare data structure: sector -> group -> value -> percentage
    sector_data = {}
    total_companies_by_sector = {}

    for sector in sector_order:
        sector_row = year_data[year_data['ateco_letter'] == sector]
        if not sector_row.empty:
            total_companies = sector_row.iloc[0]['total_companies']
            total_companies_by_sector[sector] = total_companies

            sector_dict = {}
            for group_name in field_groups.keys():
                group_dict = {}
                max_fields = field_groups[group_name]['max_fields']
                for value in range(1, max_fields + 1):  # Start from 1, excluding 0
                    count = sector_row.iloc[0][f'{group_name}_{value}']
                    percentage = (count / total_companies * 100) if total_companies > 0 else 0
                    group_dict[value] = percentage
                sector_dict[group_name] = group_dict
            sector_data[sector] = sector_dict

    # Calculate percentages for each sector
    sector_percentages = {}
    for sector in sector_order:
        if sector in sector_data:
            # Calculate total percentage for this sector (should be ≤ 100%)
            total_pct = 0
            for group_name in field_groups.keys():
                for value in range(1, field_groups[group_name]['max_fields'] + 1):
                    total_pct += sector_data[sector][group_name].get(value, 0)
            sector_percentages[sector] = min(total_pct, 100)  # Cap at 100%

    # Sort sectors by percentage (descending)
    sorted_sectors = sorted(sector_percentages.items(), key=lambda x: x[1], reverse=True)
    sorted_sector_names = [sector for sector, _ in sorted_sectors]

    # Create the stacked bars
    group_order = ['Red_Group', 'Blue_Group', 'Green_Group']

    # Initialize bottoms for each bar
    x_positions_sorted = np.arange(len(sorted_sector_names))
    bottoms = np.zeros(len(sorted_sector_names))

    # Keep track of which colors we've added to legend
    legend_added = {}

    # Calculate normalized percentages so total doesn't exceed 100%
    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Stack from 1 to max_fields (lightest to darkest)
        for value in range(1, max_fields + 1):
            # Get percentages for this value across all sectors
            percentages = []
            for i, sector in enumerate(sorted_sector_names):
                if sector in sector_data:
                    percentages.append(sector_data[sector][group_name].get(value, 0))
                else:
                    percentages.append(0)

            if sum(percentages) > 0:  # Only add if there are any
                color_idx = value - 1
                color = colors_for_group[color_idx]

                # Ensure we don't exceed 100% for any bar
                adjusted_percentages = []
                for j, pct in enumerate(percentages):
                    remaining_space = 100 - bottoms[j]
                    adjusted_percentages.append(min(pct, remaining_space))

                bars = ax.bar(x_positions_sorted, adjusted_percentages, bottom=bottoms,
                              width=bar_width, color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
                bottoms += adjusted_percentages

                # Add to legend if not already added
                legend_key = f"{group_name}_{value}"
                if legend_key not in legend_added:
                    legend_added[legend_key] = (color, value)

    # Customize the plot
    ax.set_title(f'{year} - Field Completion Distribution (Percentage)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ATECO Sector', fontsize=12)
    ax.set_ylabel('Percentage of Companies (%)', fontsize=12)

    # Set x-ticks
    ax.set_xticks(x_positions_sorted)
    ax.set_xticklabels(sorted_sector_names, rotation=45, ha='right', fontsize=11)

    # Set y-axis limit to 100%
    ax.set_ylim(0, 105)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total percentage labels on top of bars
    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(i, total + 1, f'{total:.0f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Create a comprehensive legend
    legend_items = []
    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Add group header
        legend_items.append((Line2D([0], [0], color='white'), f"<b>{group_info['name']}:</b>"))

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

        # Create legend outside the plot
        ax.legend(legend_handles, legend_labels,
                  title='Completed Fields by Group',
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=10,
                  title_fontsize=11,
                  frameon=True,
                  fancybox=True,
                  shadow=True)

    return fig, sorted_sector_names


# =============================================================================
# CREATE DISTRIBUTION VISUALIZATION
# =============================================================================

def create_distribution_chart():
    """Create a chart showing the distribution of completed fields."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution Analysis of Completed Fields (2024)', fontsize=16, fontweight='bold')
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92, hspace=0.35, wspace=0.3)

    # Data for 2024
    year = '2024'

    # 1. Overall distribution of completed fields
    ax1 = axes[0, 0]
    field_counts = df[f'total_sum_{year}'].value_counts().sort_index()
    percentages = (field_counts / len(df) * 100)

    # Sort by percentage (descending)
    sorted_indices = np.argsort(percentages.values)[::-1]
    sorted_values = percentages.values[sorted_indices]
    sorted_labels = percentages.index[sorted_indices]

    bars = ax1.bar(range(len(sorted_labels)), sorted_values, color='#3498db', alpha=0.7)
    ax1.set_title('Overall Distribution of Completed Fields', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Number of Completed Fields', fontsize=12)
    ax1.set_ylabel('Percentage of Companies (%)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='both', labelsize=11)

    # Set x-ticks
    ax1.set_xticks(range(len(sorted_labels)))
    ax1.set_xticklabels(sorted_labels, rotation=45, ha='right')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 1:
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2. Group-wise completion distribution
    ax2 = axes[0, 1]
    group_data = {}
    for group_name in field_groups.keys():
        sum_col = f'{group_name}_sum_{year}'
        group_avg = df[sum_col].mean()
        group_data[field_groups[group_name]['name']] = group_avg

    groups = list(group_data.keys())
    values = list(group_data.values())
    colors_group = ['#ff6b6b', '#4d96ff', '#6bcf7f']

    # Sort by value (descending)
    sorted_indices = np.argsort(values)[::-1]
    sorted_groups = [groups[i] for i in sorted_indices]
    sorted_values_group = [values[i] for i in sorted_indices]
    sorted_colors = [colors_group[i] for i in sorted_indices]

    bars = ax2.bar(sorted_groups, sorted_values_group, color=sorted_colors, alpha=0.7)
    ax2.set_title('Average Fields Completed by Group', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Average Number of Fields', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='both', labelsize=11)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # 3. Year-over-year comparison
    ax3 = axes[1, 0]
    years = ['2022', '2023', '2024']
    overall_avg = [df[f'total_sum_{year}'].mean() for year in years]

    # Sort years by average (they should already be in order)
    ax3.plot(years, overall_avg, marker='o', linewidth=2, markersize=8, color='#2c3e50')
    ax3.set_title('Year-over-Year Average Fields Completed', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Average Fields per Company', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(overall_avg) * 1.2 if len(overall_avg) > 0 else 10)
    ax3.tick_params(axis='both', labelsize=11)

    # Add trend line
    if len(overall_avg) > 1:
        try:
            z = np.polyfit(range(len(years)), overall_avg, 1)
            p = np.poly1d(z)
            ax3.plot(years, p(range(len(years))), "--", color='#e74c3c', alpha=0.7, label=f'Trend: {z[0]:.3f}/year')
            ax3.legend(fontsize=10)
        except:
            pass

    # 4. Sector size vs performance scatter
    ax4 = axes[1, 1]
    performance_data = calculate_comprehensive_metrics()

    if not performance_data.empty and 'overall_completion_2024' in performance_data.columns:
        sizes = performance_data['Total_Companies']
        performance = performance_data['overall_completion_2024']

        # Filter out NaN values
        valid_data = ~(sizes.isna() | performance.isna())
        if valid_data.any():
            sizes = sizes[valid_data]
            performance = performance[valid_data]

            # Sort by performance (descending) for better visualization
            sorted_indices = np.argsort(performance)[::-1]
            sizes = sizes.iloc[sorted_indices].values
            performance = performance.iloc[sorted_indices].values

            scatter = ax4.scatter(range(len(performance)), performance, alpha=0.6, c=performance,
                                  cmap='RdYlGn', s=100, edgecolor='black', linewidth=0.5)
            ax4.set_title('Sector Performance Ranking (2024)', fontweight='bold', fontsize=14)
            ax4.set_xlabel('Sector Rank', fontsize=12)
            ax4.set_ylabel('Overall Completion Rate (%)', fontsize=12)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='both', labelsize=11)

            # Set x-ticks to show sector ranks
            ax4.set_xticks(range(len(performance)))
            ax4.set_xticklabels(range(1, len(performance) + 1))

            # Add top performers labels
            for i in range(min(3, len(performance))):
                ax4.text(i, performance[i] + 2, f'{performance[i]:.1f}%',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    return fig


# =============================================================================
# CREATE ALL VISUALIZATIONS
# =============================================================================

# Create visualizations for each year
image_paths_absolute = []
image_paths_percentage = []
distribution_path = None
all_sorted_sectors = {}

print("  - Creating absolute value charts...")
for year in ['2022', '2023', '2024']:
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
        print(f"    ✗ Error creating absolute chart for {year}: {e}")
        image_paths_absolute.append(None)

print("  - Creating percentage charts...")
for year in ['2022', '2023', '2024']:
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
        print(f"    ✗ Error creating percentage chart for {year}: {e}")
        image_paths_percentage.append(None)

print("  - Creating distribution analysis chart...")
try:
    fig_dist = create_distribution_chart()
    dist_path = 'Distribution_Analysis.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    distribution_path = dist_path
    plt.close()
    print(f"    ✓ Created {dist_path}")
except Exception as e:
    print(f"    ✗ Error creating distribution chart: {e}")
    distribution_path = None

# =============================================================================
# PERFORM COMPREHENSIVE ANALYSIS
# =============================================================================

print("  - Calculating performance metrics...")
performance_analysis = calculate_comprehensive_metrics()

print("  - Calculating advanced statistics...")
advanced_stats = calculate_advanced_statistics()

print("  - Classifying sectors...")
sector_classifications = classify_sectors(performance_analysis)

# Prepare summary table for PDF
summary_data = performance_analysis[['Sector', 'Total_Companies',
                                     'overall_completion_2024',
                                     'avg_fields_per_company_2024',
                                     'pct_with_any_2024',
                                     'trend_overall_22_24']].copy()
summary_data = summary_data.sort_values('overall_completion_2024', ascending=False)
summary_data.columns = ['Sector', 'Total Cos', '2024: Overall %', '2024: Avg Fields', '2024: % Any', 'Trend 22-24']
summary_table_data = summary_data.round(1).values.tolist()
summary_table_headers = ['Sector', 'Total Cos', '2024: Overall %', '2024: Avg Fields', '2024: % Any', 'Trend 22-24']


# =============================================================================
# ENHANCED PDF REPORT WITH STATISTICAL ANALYSIS
# =============================================================================

def create_pdf_report():
    # Import colors locally to avoid conflicts
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

    # Create story (content) list
    story = []

    # Title page
    story.append(Paragraph("ATECO Sector Analysis - Enhanced Statistical Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Analysis Date: {datetime.datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph("Dataset: Tidier_Dataset.csv (16 fields grouped by color)", styles['Normal']))
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
        ['Companies with Full 16 Fields (2024)', f"{key_stats['pct_with_full']:.1f}%"],
        ['Average Fields per Company (2024)', f"{key_stats['avg_fields_per_company']:.2f}"],
        ['Median Fields per Company (2024)', f"{key_stats['median_fields']:.1f}"],
        ['Standard Deviation (2024)', f"{key_stats['std_fields']:.2f}"],
        ['Overall Trend 2022-2024', f"{trend_stats['overall_completion_growth']:+.1f}%"],
        ['Engagement Growth 2022-2024', f"{trend_stats['pct_with_any_growth']:+.1f}%"],
        ['High Performing Sectors (>40%)', f"{len(sector_classifications.get('high_performers', pd.DataFrame()))}"],
        ['Medium Performing Sectors (20-40%)',
         f"{len(sector_classifications.get('medium_performers', pd.DataFrame()))}"],
        ['Low Performing Sectors (<20%)', f"{len(sector_classifications.get('low_performers', pd.DataFrame()))}"]
    ]

    key_table = Table(key_data, colWidths=[200, 120])
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

    # Section 1: Distribution Analysis
    story.append(Paragraph("1. Statistical Distribution Analysis", heading1_style))
    story.append(Paragraph("Comprehensive analysis of field completion distributions and trends", normal_style))

    if distribution_path and os.path.exists(distribution_path):
        try:
            story.append(Image(distribution_path, width=10 * inch, height=7 * inch))
        except Exception as e:
            story.append(Paragraph(f"Could not load distribution image: {e}", normal_style))
    else:
        story.append(Paragraph("Distribution chart not available", normal_style))

    story.append(Spacer(1, 10))
    story.append(PageBreak())

    # Section 2: Absolute Value Visualizations
    story.append(Paragraph("2. Absolute Value Charts by Year", heading1_style))
    story.append(Paragraph(
        "Each bar shows the number of companies with completed fields in each sector. Sorted by total count (descending).",
        normal_style))

    for year, image_path in zip(['2022', '2023', '2024'], image_paths_absolute):
        if image_path and os.path.exists(image_path):
            story.append(Paragraph(f"{year} - Absolute Counts (Sorted by Total)", heading2_style))
            try:
                story.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                story.append(Paragraph("Image not available", normal_style))
            story.append(Spacer(1, 5))

    story.append(PageBreak())

    # Section 3: Percentage Visualizations
    story.append(Paragraph("3. Percentage Charts by Year", heading1_style))
    story.append(Paragraph(
        "Each bar shows the percentage of companies with completed fields in each sector. Bars capped at 100%. Sorted by percentage (descending).",
        normal_style))

    for year, image_path in zip(['2022', '2023', '2024'], image_paths_percentage):
        if image_path and os.path.exists(image_path):
            story.append(Paragraph(f"{year} - Percentage Distribution (Sorted by %)", heading2_style))
            try:
                story.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                story.append(Paragraph("Image not available", normal_style))
            story.append(Spacer(1, 5))

    story.append(PageBreak())

    # Section 4: Sector Performance Classification
    story.append(Paragraph("4. Sector Performance Classification", heading1_style))

    # High Performers
    high_performers = sector_classifications.get('high_performers', pd.DataFrame())
    if len(high_performers) > 0:
        story.append(
            Paragraph(f"High Performers (>40% overall completion): {len(high_performers)} sectors", heading2_style))
        for _, row in high_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg completion, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend: {row.get('trend_overall_22_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No high performing sectors found", normal_style))

    # Medium Performers
    medium_performers = sector_classifications.get('medium_performers', pd.DataFrame())
    if len(medium_performers) > 0:
        story.append(Paragraph(f"Medium Performers (20-40% overall completion): {len(medium_performers)} sectors",
                               heading2_style))
        for _, row in medium_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg completion, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend: {row.get('trend_overall_22_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No medium performing sectors found", normal_style))

    # Low Performers
    low_performers = sector_classifications.get('low_performers', pd.DataFrame())
    if len(low_performers) > 0:
        story.append(
            Paragraph(f"Low Performers (<20% overall completion): {len(low_performers)} sectors", heading2_style))
        for _, row in low_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg completion, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend: {row.get('trend_overall_22_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No low performing sectors found", normal_style))

    story.append(PageBreak())

    # Section 5: Performance Summary Table
    story.append(Paragraph("5. Performance Summary Table (2024)", heading1_style))
    story.append(Paragraph("All sectors sorted by overall completion percentage in 2024 (descending)", normal_style))

    # Create summary table
    if len(summary_table_data) > 0:
        summary_data_table = [summary_table_headers] + summary_table_data

        summary_table = Table(summary_data_table, colWidths=[60, 70, 80, 80, 80, 70])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))

        story.append(summary_table)
    else:
        story.append(Paragraph("No summary data available", normal_style))

    story.append(Spacer(1, 20))

    # Section 6: Overall Engagement Trends
    story.append(Paragraph("6. Overall Engagement Trends (All Sectors Combined)", heading1_style))

    overall_data = [
        ['Year', 'Overall Completion %', 'Companies with Any Data %', 'Avg Fields/Company', 'Trend'],
        ['2022', f"{advanced_stats['yearly_stats']['2022']['overall_completion']:.1f}%",
         f"{advanced_stats['yearly_stats']['2022']['pct_with_any']:.1f}%",
         f"{advanced_stats['yearly_stats']['2022']['avg_fields_per_company']:.2f}", '-'],
        ['2023', f"{advanced_stats['yearly_stats']['2023']['overall_completion']:.1f}%",
         f"{advanced_stats['yearly_stats']['2023']['pct_with_any']:.1f}%",
         f"{advanced_stats['yearly_stats']['2023']['avg_fields_per_company']:.2f}",
         f"{advanced_stats['yearly_stats']['2023']['overall_completion'] - advanced_stats['yearly_stats']['2022']['overall_completion']:+.1f}%"],
        ['2024', f"{advanced_stats['yearly_stats']['2024']['overall_completion']:.1f}%",
         f"{advanced_stats['yearly_stats']['2024']['pct_with_any']:.1f}%",
         f"{advanced_stats['yearly_stats']['2024']['avg_fields_per_company']:.2f}",
         f"{advanced_stats['yearly_stats']['2024']['overall_completion'] - advanced_stats['yearly_stats']['2023']['overall_completion']:+.1f}%"],
        ['Overall Trend (2022-2024)', f"{advanced_stats['trend_stats']['overall_completion_growth']:+.1f}%",
         f"{advanced_stats['trend_stats']['pct_with_any_growth']:+.1f}%",
         f"{advanced_stats['trend_stats']['avg_fields_growth']:+.2f}",
         f"{advanced_stats['trend_stats']['overall_completion_growth']:+.1f}%"]
    ]

    overall_table = Table(overall_data, colWidths=[100, 120, 120, 120, 80])
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

    # Section 7: Statistical Insights
    story.append(Paragraph("7. Advanced Statistical Insights", heading1_style))

    insights = [
        f"1. Distribution Analysis: The completion distribution shows {('high' if advanced_stats['yearly_stats']['2024']['std_fields'] > 3 else 'moderate' if advanced_stats['yearly_stats']['2024']['std_fields'] > 1.5 else 'low')} variability across companies.",
        f"2. Central Tendency: Median completion ({advanced_stats['yearly_stats']['2024']['median_fields']:.1f} fields) vs Mean ({advanced_stats['yearly_stats']['2024']['avg_fields_per_company']:.2f} fields) indicates {('right-skewed' if advanced_stats['yearly_stats']['2024']['median_fields'] < advanced_stats['yearly_stats']['2024']['avg_fields_per_company'] else 'left-skewed' if advanced_stats['yearly_stats']['2024']['median_fields'] > advanced_stats['yearly_stats']['2024']['avg_fields_per_company'] else 'symmetric')} distribution.",
        f"3. Quartile Analysis: 25% of companies complete ≤{advanced_stats['yearly_stats']['2024']['q1_fields']:.1f} fields, while 75% complete ≤{advanced_stats['yearly_stats']['2024']['q3_fields']:.1f} fields.",
        f"4. Annual Growth: Average annual growth rate in completion is {advanced_stats['trend_stats']['overall_completion_growth'] / 2:.2f}% per year (2022-2024).",
        f"5. Engagement Rate: {advanced_stats['yearly_stats']['2024']['pct_with_any']:.1f}% of companies have at least one completed field.",
        f"6. Full Compliance: Only {advanced_stats['yearly_stats']['2024']['pct_with_full']:.1f}% of companies have all 16 fields completed.",
        "7. Visual Analysis: Charts are sorted in descending order for easier comparison of sector performance."
    ]

    for insight in insights:
        story.append(Paragraph(insight, bullet_style))

    # Conclusions
    story.append(PageBreak())
    story.append(Paragraph("Conclusions and Strategic Recommendations", heading1_style))

    conclusions = [
        "1. Focus improvement efforts on sectors with consistently low completion rates across all field groups.",
        "2. Develop targeted strategies for each field group based on sector-specific performance patterns.",
        "3. Monitor sectors showing declining trends for early intervention and support.",
        "4. Recognize and replicate best practices from high-performing sectors with similar characteristics.",
        "5. Use both absolute and percentage views to understand both scale and proportional engagement.",
        "6. Implement group-specific training programs for sectors struggling with particular field categories.",
        "7. Set realistic improvement targets based on quartile analysis and distribution patterns.",
        "8. Regular statistical monitoring should inform resource allocation and strategy adjustments.",
        "9. Consider sector size when designing engagement strategies, as challenges differ between large and small sectors."
    ]

    for conclusion in conclusions:
        story.append(Paragraph(conclusion, bullet_style))

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

if __name__ == "__main__":
    try:
        print(f"\n✓ Analysis complete!")

        # Generate PDF report
        print(f"✓ Generating enhanced PDF report...")
        pdf_file = create_pdf_report()

        if pdf_file:
            print(f"✓ Enhanced PDF report successfully created: {pdf_file}")
        else:
            print("✗ Failed to create PDF report")

        print(f"\n✓ Statistical Summary:")
        if 'yearly_stats' in advanced_stats and '2024' in advanced_stats['yearly_stats']:
            key_stats = advanced_stats['yearly_stats']['2024']
            print(f"  - Overall completion (2024): {key_stats['overall_completion']:.1f}%")
            print(f"  - Companies with any data (2024): {key_stats['pct_with_any']:.1f}%")
            print(f"  - Average fields per company (2024): {key_stats['avg_fields_per_company']:.2f}")
            print(f"  - Standard deviation (2024): {key_stats['std_fields']:.2f}")

        if 'trend_stats' in advanced_stats:
            print(f"  - Overall trend 2022-2024: {advanced_stats['trend_stats']['overall_completion_growth']:+.1f}%")

        print(f"  - High performing sectors: {len(sector_classifications.get('high_performers', pd.DataFrame()))}")
        print(f"  - Medium performing sectors: {len(sector_classifications.get('medium_performers', pd.DataFrame()))}")
        print(f"  - Low performing sectors: {len(sector_classifications.get('low_performers', pd.DataFrame()))}")

        print(f"\n✓ Image files created:")
        valid_abs = [p for p in image_paths_absolute if p is not None]
        valid_pct = [p for p in image_paths_percentage if p is not None]
        print(f"  - Absolute charts: {len(valid_abs)} created")
        print(f"  - Percentage charts: {len(valid_pct)} created")
        if distribution_path:
            print(f"  - Distribution Analysis: {distribution_path}")

    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        print(f"Error details: {traceback.format_exc()}")