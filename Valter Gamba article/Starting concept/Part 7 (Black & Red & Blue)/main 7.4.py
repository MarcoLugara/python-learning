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
        'color_base': '#ff9800',  # Changed from '#ff6b6b' to '#ff9800' (orange/amber)
        'name': 'Fields 1-8',
        'max_fields': 8
    },
    'Blue_Group': {
        'fields': [f'Field{i}' for i in range(9, 14)],  # Fields 9-13
        'color_base': '#4d96ff',  # Blue base - unchanged
        'name': 'Fields 9-13',
        'max_fields': 5
    },
    'Green_Group': {
        'fields': [f'Field{i}' for i in range(14, 17)],  # Fields 14-16
        'color_base': '#6bcf7f',  # Green base - unchanged
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

    # Generate shades - ADJUSTED FOR BETTER CONTRAST
    shades = []
    if reverse:
        # For better contrast in reverse, use wider range
        light_values = np.linspace(0.85, 0.25, num_shades)  # Much lighter to darker
    else:
        # For better contrast in normal direction, use wider range
        light_values = np.linspace(0.25, 0.85, num_shades)  # Much darker to lighter

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
# CREATE VISUALIZATIONS - ABSOLUTE VALUES (FIXED)
# =============================================================================

def create_absolute_bar_chart(year_data, year, sector_order):
    """Create a bar chart with absolute counts (excluding zero values)."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # ADJUSTED: Optimized bottom margin for better spacing
    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.18, top=0.92)

    # Set up bar positions and width
    x_positions = np.arange(len(sector_order))
    bar_width = 0.7

    # Prepare data structure: sector -> group -> value -> count
    sector_data = {}
    sector_totals = {}  # Store total companies per sector
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

    # Recreate the figure with sorted data
    fig, ax = plt.subplots(figsize=(16, 8))

    # ADJUSTED: Reduced bottom margin slightly and increased figure height
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

    # ADJUSTED: Fixed vertical positioning of sector labels
    ax.set_xticklabels(sorted_sectors, rotation=45, ha='right', fontsize=12)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total count labels on top of bars
    max_height = max(bottoms_sorted) if len(bottoms_sorted) > 0 else 1
    for i, total in enumerate(bottoms_sorted):
        if total > 0:
            ax.text(i, total + max_height * 0.01, f'{int(total)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ADDED: Add 'n' labels at the bottom with MUCH CLOSER spacing
    y_min, y_max = ax.get_ylim()

    # Calculate SMALLER offset to bring 'n' labels much closer
    if max_height > 0:
        offset = max_height * 0.05  # Reduced from 0.15 to 0.05 (much closer)
    else:
        offset = 0.5  # fallback

    # Set y-limit to accommodate 'n' labels with very close spacing
    ax.set_ylim(bottom=y_min - offset)

    # Calculate position VERY CLOSE to x-axis (similar to distance between letter and n)
    n_label_y_position = y_min - offset * 0.3  # Reduced from 0.7 to 0.3 (much closer)

    # Add 'n' labels with VERY CLOSE vertical position
    for i, sector in enumerate(sorted_sectors):
        total_companies = sorted_sector_totals.get(sector, 0)
        ax.text(i, n_label_y_position, f'n={total_companies}',
                ha='center', va='top', fontsize=9, color='gray', fontweight='bold')

    # Create a comprehensive legend (REMOVED HTML TAGS)
    legend_items = []
    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Add group header (REMOVED <b> tags)
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
# CREATE VISUALIZATIONS - PERCENTAGES (FIXED - ADJUSTED FOR REALISTIC %)
# =============================================================================

def create_percentage_bar_chart(year_data, year, sector_order):
    """Create a stacked bar chart showing percentage of total fields completed, stratified by group and field count."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # ADJUSTED: Optimized bottom margin for better spacing
    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.18, top=0.92)

    # Calculate percentages for each sector, stratified by group and field count
    sector_stratified_data = {}
    sector_totals = {}
    sector_total_percentages = {}

    for sector in sector_order:
        sector_row = year_data[year_data['ateco_letter'] == sector]
        if not sector_row.empty:
            total_companies = sector_row.iloc[0]['total_companies']
            sector_totals[sector] = total_companies

            # Calculate total possible fields (16 fields per company)
            total_possible_fields = 16 * total_companies if total_companies > 0 else 1

            # For each group and value (1 to max_fields), calculate contribution percentage
            stratified_dict = {}
            total_percentage = 0

            for group_name, group_info in field_groups.items():
                group_dict = {}
                max_fields = group_info['max_fields']

                for value in range(1, max_fields + 1):  # Start from 1, exclude 0
                    count = sector_row.iloc[0].get(f'{group_name}_{value}', 0)
                    # Each company with 'value' completed fields in this group contributes:
                    # (value * count) fields completed
                    fields_completed = value * count
                    # Percentage contribution to total possible fields
                    percentage = (fields_completed / total_possible_fields * 100) if total_possible_fields > 0 else 0
                    group_dict[value] = percentage
                    total_percentage += percentage

                stratified_dict[group_name] = group_dict

            sector_stratified_data[sector] = stratified_dict
            sector_total_percentages[sector] = total_percentage

    # Sort sectors by total percentage (descending)
    sorted_sectors = sorted(sector_total_percentages.items(), key=lambda x: x[1], reverse=True)
    sorted_sector_names = [sector for sector, _ in sorted_sectors]

    # Create the stacked bars - same stratification as absolute chart
    group_order = ['Red_Group', 'Blue_Group', 'Green_Group']

    # Initialize bottoms for each bar
    x_positions_sorted = np.arange(len(sorted_sector_names))
    bottoms = np.zeros(len(sorted_sector_names))

    # Keep track of which colors/values we've added to legend
    legend_added = {}

    # Stack in the same order as absolute chart: Red (1-8), Blue (1-5), Green (1-3)
    # For visual clarity, we'll stack from lowest to highest value within each group
    for group_name in group_order:
        group_info = field_groups[group_name]
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Stack from 1 to max_fields (lightest to darkest)
        for value in range(1, max_fields + 1):
            # Get percentages for this value across all sorted sectors
            percentages = []
            for sector in sorted_sector_names:
                if sector in sector_stratified_data:
                    percentages.append(sector_stratified_data[sector][group_name].get(value, 0))
                else:
                    percentages.append(0)

            if sum(percentages) > 0:  # Only add if there are any
                color_idx = value - 1
                color = colors_for_group[color_idx] if color_idx < len(colors_for_group) else colors_for_group[-1]

                # Plot this segment
                ax.bar(x_positions_sorted, percentages, bottom=bottoms,
                       width=0.7, color=color, alpha=0.9,
                       edgecolor='black', linewidth=0.5)
                bottoms += percentages

                # Add to legend if not already added
                legend_key = f"{group_name}_{value}"
                if legend_key not in legend_added:
                    legend_added[legend_key] = (color, group_name, value)

    # Customize the plot
    ax.set_title(f'{year} - Percentage of Total Fields Completed (Stratified)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ATECO Sector', fontsize=12)
    ax.set_ylabel('Percentage of Total Possible Fields Completed (%)', fontsize=12)

    # Set x-ticks with better positioning
    ax.set_xticks(x_positions_sorted)

    # ADJUSTED: Fixed vertical positioning of sector labels
    ax.set_xticklabels(sorted_sector_names, rotation=45, ha='right', fontsize=12)

    # Set y-axis limit based on highest column plus padding (but capped at 100)
    max_bar_height = max(bottoms) if len(bottoms) > 0 else 0
    # Add 10% padding, but don't exceed 100%
    y_max = min(100, max_bar_height * 1.1)
    # Ensure we have at least some space for labels if all bars are 0
    if y_max < 10:
        y_max = 10

    # ADJUSTED: Use SMALLER offset calculation for closer spacing
    y_min = 0
    offset = y_max * 0.05  # Reduced from 0.15 to 0.05 (much closer)

    ax.set_ylim(bottom=y_min - offset, top=y_max)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total percentage labels on top of bars
    for i, (sector, total_pct) in enumerate(zip(sorted_sector_names, bottoms)):
        if total_pct > 0:
            # Simple percentage label without box
            ax.text(i, total_pct + y_max * 0.01, f'{int(total_pct)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add 'n' labels at the bottom with MUCH CLOSER spacing
    # Calculate position VERY CLOSE to x-axis
    n_label_y_position = y_min - offset * 0.3  # Reduced from 0.7 to 0.3 (much closer)

    for i, sector in enumerate(sorted_sector_names):
        total_companies = sector_totals.get(sector, 0)
        # All 'n' labels at same vertical position, MUCH CLOSER
        ax.text(i, n_label_y_position, f'n={total_companies}',
                ha='center', va='top', fontsize=9, color='gray', fontweight='bold')

    # Create a comprehensive legend (REMOVED HTML TAGS)
    legend_items = []

    # First, create group headers (REMOVED <b> tags)
    for group_name in group_order:
        group_info = field_groups[group_name]
        legend_items.append((Line2D([0], [0], color='white'),
                             f"{group_info['name']}:"))

        # Then add value items for this group
        max_fields = group_info['max_fields']
        colors_for_group = group_colors[group_name]

        # Show from 1 to max_fields
        for value in range(1, max_fields + 1):
            legend_key = f"{group_name}_{value}"
            if legend_key in legend_added:
                color, _, val = legend_added[legend_key]
                legend_items.append((Patch(facecolor=color, edgecolor='black',
                                           alpha=0.9, linewidth=0.5),
                                     f"  {val} field{'s' if val > 1 else ''}"))

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
# CREATE ALL VISUALIZATIONS
# =============================================================================

# Create visualizations for each year
image_paths_absolute = []
image_paths_percentage = []
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
        print(f"    ✗ Error creating absolute chart for {year}: {str(e)[:100]}...")
        import traceback

        print(f"    Detailed error: {traceback.format_exc()[:200]}...")
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
# ENHANCED PDF REPORT WITH STATISTICAL ANALYSIS (NO DISTRIBUTION GRAPH)
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

    # Create a style for table headers
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
    story.append(Paragraph("Dataset: Starting_Dataset.csv (16 fields grouped by color)", styles['Normal']))
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

    # WIDER columns to prevent overlap
    key_table = Table(key_data, colWidths=[240, 160])
    key_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),  # Reduced font size
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),  # Reduced font size
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

    # Section 1: Distribution Analysis (Text only, no graph)
    story.append(Paragraph("1. Statistical Distribution Analysis", heading1_style))
    story.append(
        Paragraph("This section shows how many companies completed different numbers of fields (0-16).", normal_style))

    # Distribution statistics table
    dist_headers = ['Fields', 'Companies (2022)', '%', 'Companies (2023)', '%', 'Companies (2024)', '%']
    dist_data = [dist_headers]

    # Show only key distributions (0, 1-5, 6-10, 11-15, 16)
    ranges = [
        (0, 0, "0"),
        (1, 5, "1-5"),
        (6, 10, "6-10"),
        (11, 15, "11-15"),
        (16, 16, "16")
    ]

    for start, end, label in ranges:
        row = [label]
        for year in ['2022', '2023', '2024']:
            year_dist = advanced_stats['distribution_data'][year]
            if start == end:
                # Single value
                if start < len(year_dist):
                    companies = year_dist[start]['companies']
                    percentage = year_dist[start]['percentage']
                    row.append(f"{companies:,}")
                    row.append(f"{percentage:.1f}%")
                else:
                    row.append("0")
                    row.append("0.0%")
            else:
                # Range
                total_companies = 0
                total_pct = 0
                for i in range(start, end + 1):
                    if i < len(year_dist):
                        total_companies += year_dist[i]['companies']
                        total_pct += year_dist[i]['percentage']
                row.append(f"{total_companies:,}")
                row.append(f"{total_pct:.1f}%")
        dist_data.append(row)

    # WIDER columns for distribution table
    dist_table = Table(dist_data, colWidths=[90, 90, 80, 90, 80, 90, 80])
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

    for year, image_path in zip(['2022', '2023', '2024'], image_paths_absolute):
        if image_path and os.path.exists(image_path):
            # Use KeepTogether to ensure blue title and image stay on same page
            content_to_keep_together = []
            content_to_keep_together.append(Paragraph(f"{year} - Absolute Counts (Sorted by Total)", heading2_style))
            try:
                content_to_keep_together.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                content_to_keep_together.append(Paragraph("Image not available", normal_style))
            content_to_keep_together.append(Spacer(1, 5))

            # Wrap the heading and image in KeepTogether
            story.append(KeepTogether(content_to_keep_together))

    story.append(PageBreak())

    # Section 3: Percentage Visualizations
    story.append(Paragraph("3. Percentage Charts by Year", heading1_style))
    story.append(Paragraph(
        "Each bar shows what percentage of the total possible fields were completed in each sector. Sorted by percentage (descending).",
        normal_style))
    story.append(Paragraph("How to read: Taller bars = higher percentage of fields completed", bullet_style))
    story.append(Paragraph("Note: 100% would mean ALL companies completed ALL 16 fields", bullet_style))
    story.append(Paragraph("Color meaning: Same as absolute charts (see legend)", bullet_style))
    story.append(Paragraph("n= shows total number of companies in that sector", bullet_style))

    for year, image_path in zip(['2022', '2023', '2024'], image_paths_percentage):
        if image_path and os.path.exists(image_path):
            # Use KeepTogether to ensure blue title and image stay on same page
            content_to_keep_together = []
            content_to_keep_together.append(
                Paragraph(f"{year} - Percentage Distribution (Sorted by %)", heading2_style))
            try:
                content_to_keep_together.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                content_to_keep_together.append(Paragraph("Image not available", normal_style))
            content_to_keep_together.append(Spacer(1, 5))

            # Wrap the heading and image in KeepTogether
            story.append(KeepTogether(content_to_keep_together))

    story.append(PageBreak())

    # Section 4: Sector Performance Classification
    story.append(Paragraph("4. Sector Performance Classification", heading1_style))
    story.append(Paragraph("Note: Classification based on average performance across 2022-2024", normal_style))
    story.append(Spacer(1, 10))

    # High Performers
    high_performers = sector_classifications.get('high_performers', pd.DataFrame())
    if len(high_performers) > 0:
        story.append(
            Paragraph(f"High Performers (>40% overall completion): {len(high_performers)} sectors", heading2_style))
        story.append(Paragraph("These sectors consistently completed the highest percentage of fields:", normal_style))
        for _, row in high_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg completion, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend (2022-2024): {row.get('trend_overall_22_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No high performing sectors found", normal_style))

    story.append(Spacer(1, 10))

    # Medium Performers
    medium_performers = sector_classifications.get('medium_performers', pd.DataFrame())
    if len(medium_performers) > 0:
        story.append(Paragraph(f"Medium Performers (20-40% overall completion): {len(medium_performers)} sectors",
                               heading2_style))
        story.append(Paragraph("These sectors completed a moderate percentage of fields:", normal_style))
        for _, row in medium_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg completion, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend (2022-2024): {row.get('trend_overall_22_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No medium performing sectors found", normal_style))

    story.append(Spacer(1, 10))

    # Low Performers
    low_performers = sector_classifications.get('low_performers', pd.DataFrame())
    if len(low_performers) > 0:
        story.append(
            Paragraph(f"Low Performers (<20% overall completion): {len(low_performers)} sectors", heading2_style))
        story.append(Paragraph("These sectors completed the lowest percentage of fields:", normal_style))
        for _, row in low_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg completion, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend (2022-2024): {row.get('trend_overall_22_24', 0):+.1f}%",
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
    story.append(Paragraph("• Trend 22-24: Change in overall completion from 2022 to 2024 (positive = improvement)",
                           bullet_style))
    story.append(Spacer(1, 10))

    # Create summary table
    if len(summary_table_data) > 0:
        summary_data_table = [summary_table_headers] + summary_table_data

        # WIDER columns for summary table
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
    story.append(Paragraph("This table shows how overall completion has changed from 2022 to 2024:", normal_style))

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

    # WIDER columns for overall table - EXTRA WIDE to prevent overlap
    overall_table = Table(overall_data, colWidths=[120, 140, 150, 140, 100])
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, 3), colors.HexColor('#f8f9fa')),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'),
    ]))

    story.append(overall_table)
    story.append(Spacer(1, 15))

    # Add simple explanation of trends
    story.append(Paragraph("Trend Interpretation:", heading2_style))
    if trend_stats['overall_completion_growth'] > 0:
        story.append(Paragraph(
            f"✓ Overall completion increased by {trend_stats['overall_completion_growth']:.1f}% from 2022 to 2024",
            bullet_style))
    else:
        story.append(Paragraph(
            f"✗ Overall completion decreased by {abs(trend_stats['overall_completion_growth']):.1f}% from 2022 to 2024",
            bullet_style))

    if trend_stats['pct_with_any_growth'] > 0:
        story.append(
            Paragraph(f"✓ More companies now have at least some data (+{trend_stats['pct_with_any_growth']:.1f}%)",
                      bullet_style))
    else:
        story.append(Paragraph(f"✗ Fewer companies now have any data (-{abs(trend_stats['pct_with_any_growth']):.1f}%)",
                               bullet_style))

    story.append(PageBreak())

    # Section 7: Key Findings and Simple Insights
    story.append(Paragraph("7. Key Findings and Simple Insights", heading1_style))

    insights = [
        f"1. Overall Performance: In 2024, companies completed {key_stats['overall_completion']:.1f}% of all possible fields.",
        f"2. Company Participation: {key_stats['pct_with_any']:.1f}% of companies completed at least 1 field in 2024.",
        f"3. Full Compliance: Only {key_stats['pct_with_full']:.1f}% of companies completed all 16 fields in 2024.",
        f"4. Average Completion: Each company completed an average of {key_stats['avg_fields_per_company']:.1f} fields in 2024.",
        f"5. Most Common: The most common number of completed fields is {mode_str}.",
        f"6. Yearly Growth: From 2022 to 2024, completion increased by {trend_stats['overall_completion_growth']:+.1f}%.",
        f"7. Best Performing Sectors: {len(sector_classifications.get('high_performers', pd.DataFrame()))} sectors consistently completed >40% of fields.",
        f"8. Areas for Improvement: {len(sector_classifications.get('low_performers', pd.DataFrame()))} sectors completed <20% of fields."
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

        print(f"\n✓ Key Fixes Applied:")
        print(f"  1. Brought 'n' labels MUCH closer to bars (reduced offset from 0.15 to 0.05)")
        print(f"  2. Removed <b> and </b> tags from image legends")
        print(f"  3. Removed 'Suggested Focus Areas' section")
        print(f"  4. Used KeepTogether to ensure graph titles and images stay on same page")

    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        print(f"Error details: {traceback.format_exc()}")