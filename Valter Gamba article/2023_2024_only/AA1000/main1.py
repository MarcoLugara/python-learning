import pandas as pd

######################################


# Alternative version that's more dynamic
def create_dataset(input_file, output_file1, output_file2):
    """
    Create a dataset with only 2023 and 2024 data from the original dataset.

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path where the output CSV will be saved
    """

    # Read the original dataset
    df = pd.read_csv(input_file)

    # Remove columns that contain '2022' in their name
    columns_to_keep = [col for col in df.columns if '2022' not in col]

    # Create the new dataframe
    df = df[columns_to_keep].copy()

    # Save to CSV and Excel
    df.to_csv(output_file1, index=False)
    df.to_excel(output_file2, index=False)

    return df

#OLD CODE
'''
# Create the dataset
df = create_dataset('Starting_Dataset.csv', 'Tidier_Dataset.csv', 'Tidier_Dataset.xlsx')

'''

#CODE
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

# Load the data
df = pd.read_csv('Tidier_Dataset.csv')

# Extract ATECO letter from the 'ateco' column
df['ateco_letter'] = df['ateco'].str[0]

# Define field groups - SINGLE GROUP with 2 years (2023, 2024)
field_groups = {
    'AA1000_Group': {
        'fields': ['AA1000'],  # Single field
        'color_base': '#3498db',  # Blue color
        'name': 'AA1000',
        'max_fields': 1,  # Each field is binary (0 or 1)
        'years': ['2023', '2024']  # Only 2 years
    }
}

# Calculate sum of ones for each group and year
for group_name, group_info in field_groups.items():
    for year in group_info['years']:
        field_columns = [f'{field}_{year}' for field in group_info['fields']]
        df[f'{group_name}_sum_{year}'] = df[field_columns].sum(axis=1)

# Calculate overall sum for each year
for year in field_groups['AA1000_Group']['years']:
    all_field_columns = [f'AA1000_{year}']
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


# Generate color gradients for the group
group_colors = {}
for group_name, group_info in field_groups.items():
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
            max_possible = 1 * total_companies  # Only 1 field per year
            overall_completion = (total_fields_completed / max_possible * 100) if max_possible > 0 else 0

            sector_metrics[f'overall_completion_{year}'] = overall_completion
            sector_metrics[
                f'avg_fields_per_company_{year}'] = total_fields_completed / total_companies if total_companies > 0 else 0

            # Percentage with any completion (AA1000 = 1)
            companies_with_any = len(sector_data[sector_data[f'total_sum_{year}'] > 0])
            sector_metrics[f'pct_with_any_{year}'] = (
                        companies_with_any / total_companies * 100) if total_companies > 0 else 0

            # Percentage with full completion (1 field)
            companies_with_full = len(sector_data[sector_data[f'total_sum_{year}'] == 1])
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
            'companies_with_full': len(df[df[f'total_sum_{year}'] == 1]),  # Full = 1 field
            'total_fields_completed': df[f'total_sum_{year}'].sum(),
            'max_possible_fields': len(df) * 1  # Only 1 field per year
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
        for count in range(0, 2):  # 0 to 1 field (binary)
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

    return {
        'yearly_stats': stats_dict,
        'distribution_data': distribution_data,
        'trend_stats': trend_stats
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
# CREATE VISUALIZATIONS - ABSOLUTE VALUES
# =============================================================================

def create_absolute_bar_chart(year_data, year, sector_order):
    """Create a bar chart with absolute counts - SHOWING ONLY ADOPTED COMPANIES."""
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
                for value in range(0, max_fields + 1):  # 0 and 1
                    count = sector_row.iloc[0][f'{group_name}_{value}']
                    group_dict[value] = count
                sector_dict[group_name] = group_dict
            sector_data[sector] = sector_dict

    # Create the bars - ONLY SHOW VALUE 1 (ADOPTED)
    group_order = ['AA1000_Group']

    # Colors - ONLY GREEN FOR ADOPTED
    color_adopted = '#2ecc71'  # Green for adopted

    # Get counts for ADOPTED (1) only
    adopted_counts = []
    for i, sector in enumerate(sector_order):
        if sector in sector_data:
            adopted_counts.append(sector_data[sector]['AA1000_Group'].get(1, 0))
        else:
            adopted_counts.append(0)

    # Calculate total heights for sorting (based on adopted counts only)
    total_adopted = np.array(adopted_counts)

    # Sort sectors by number of adopters (descending)
    sorted_indices = np.argsort(total_adopted)[::-1]
    sorted_sectors = [sector_order[i] for i in sorted_indices]

    # Recreate the figure with sorted data
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.18, top=0.92)

    x_positions_sorted = np.arange(len(sorted_sectors))

    # Recreate sorted sector data
    sorted_sector_data = {}
    sorted_sector_totals = {}
    sorted_adopted_counts = []

    for sector in sorted_sectors:
        if sector in sector_data:
            sorted_sector_data[sector] = sector_data[sector]
            sorted_sector_totals[sector] = sector_totals[sector]
            sorted_adopted_counts.append(sector_data[sector]['AA1000_Group'].get(1, 0))
        else:
            sorted_adopted_counts.append(0)

    # Plot ONLY ADOPTED counts (green bars)
    bars = ax.bar(x_positions_sorted, sorted_adopted_counts,
                  width=bar_width, color=color_adopted, alpha=0.9,
                  edgecolor='black', linewidth=0.5)

    # Customize the plot
    ax.set_title(f'AA1000 Adoption - {year} (Number of Companies that Adopted)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ATECO Sector', fontsize=12)
    ax.set_ylabel('Number of Companies with AA1000', fontsize=12)

    ax.set_xticks(x_positions_sorted)
    ax.set_xticklabels(sorted_sectors, rotation=45, ha='right', fontsize=12)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add count labels on top of bars
    max_height = max(sorted_adopted_counts) if len(sorted_adopted_counts) > 0 else 1
    for i, count in enumerate(sorted_adopted_counts):
        if count > 0:
            ax.text(i, count + max_height * 0.01, f'{int(count)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add 'n' labels at the bottom (total companies in sector)
    y_min, y_max = ax.get_ylim()

    if max_height > 0:
        offset = max_height * 0.15
    else:
        offset = 0.5

    # Set y-axis limit to max value plus 10%
    ax.set_ylim(bottom=y_min - offset, top=max_height * 1.10)

    n_label_y_position = y_min - offset * 0.7

    for i, sector in enumerate(sorted_sectors):
        total_companies = sorted_sector_totals.get(sector, 0)
        ax.text(i, n_label_y_position, f'n={total_companies}',
                ha='center', va='top', fontsize=9, color='gray', fontweight='bold')

    # Create simplified legend (only adopted)
    legend_items = [
        (Patch(facecolor='#2ecc71', edgecolor='black', alpha=0.9, linewidth=0.5),
         "AA1000 Adopted")
    ]

    legend_handles, legend_labels = zip(*legend_items)

    ax.legend(legend_handles, legend_labels,
              title='Status',
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
    """Create a stacked bar chart showing percentage of companies with AA1000."""
    fig, ax = plt.subplots(figsize=(16, 8))

    plt.subplots_adjust(left=0.08, right=0.75, bottom=0.18, top=0.92)

    # Calculate percentages for each sector
    sector_percentages = {}
    sector_totals = {}

    for sector in sector_order:
        sector_row = year_data[year_data['ateco_letter'] == sector]
        if not sector_row.empty:
            total_companies = sector_row.iloc[0]['total_companies']
            sector_totals[sector] = total_companies

            # Get count of companies with AA1000 = 1
            count_aa1000 = sector_row.iloc[0].get('AA1000_Group_1', 0)
            percentage_aa1000 = (count_aa1000 / total_companies * 100) if total_companies > 0 else 0

            sector_percentages[sector] = percentage_aa1000

    # Sort sectors by percentage (descending)
    sorted_sectors = sorted(sector_percentages.items(), key=lambda x: x[1], reverse=True)
    sorted_sector_names = [sector for sector, _ in sorted_sectors]
    sorted_percentages = [percentage for _, percentage in sorted_sectors]

    # Create the bars
    x_positions_sorted = np.arange(len(sorted_sector_names))

    # Colors: green for AA1000 adopted, gray for not adopted (background)
    colors_for_bars = ['#2ecc71' for _ in sorted_percentages]  # All green for adoption rate

    # Create bars (showing only the adoption percentage)
    bars = ax.bar(x_positions_sorted, sorted_percentages,
                  width=0.7, color=colors_for_bars, alpha=0.9,
                  edgecolor='black', linewidth=0.5)

    # Customize the plot
    ax.set_title(f'AA1000 Adoption Rate - {year}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('ATECO Sector', fontsize=12)
    ax.set_ylabel('Percentage of Companies with AA1000 (%)', fontsize=12)

    ax.set_xticks(x_positions_sorted)
    ax.set_xticklabels(sorted_sector_names, rotation=45, ha='right', fontsize=12)

    # Set y-axis limit to max value plus 10% (not fixed at 100%)
    max_percentage = max(sorted_percentages) if len(sorted_percentages) > 0 else 1
    ax.set_ylim(0, min(100, max_percentage * 1.10))  # Cap at 100% if needed
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels on top of bars
    for i, (sector, pct) in enumerate(zip(sorted_sector_names, sorted_percentages)):
        if pct > 0:
            ax.text(i, pct + (max_percentage * 0.01), f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add 'n' labels at the bottom
    y_min, y_max = ax.get_ylim()
    offset = max_percentage * 0.10 if max_percentage > 0 else 10  # Dynamic offset based on max percentage

    ax.set_ylim(bottom=y_min - offset)

    n_label_y_position = y_min - offset * 0.5

    for i, sector in enumerate(sorted_sector_names):
        total_companies = sector_totals.get(sector, 0)
        ax.text(i, n_label_y_position, f'n={total_companies}',
                ha='center', va='top', fontsize=9, color='gray', fontweight='bold')

    # Create legend
    legend_items = [
        (Patch(facecolor='#2ecc71', edgecolor='black', alpha=0.9, linewidth=0.5),
         "AA1000 Adoption Rate")
    ]

    legend_handles, legend_labels = zip(*legend_items)

    ax.legend(legend_handles, legend_labels,
              title='Metric',
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

# Prepare summary table for PDF
summary_data = performance_analysis[['Sector', 'Total_Companies',
                                     'overall_completion_2024',
                                     'avg_fields_per_company_2024',
                                     'pct_with_any_2024',
                                     'trend_overall_23_24']].copy()
summary_data = summary_data.sort_values('overall_completion_2024', ascending=False)
summary_data.columns = ['Sector', 'Total Cos', '2024: AA1000 %', '2024: Avg Fields', '2024: % Any', 'Trend 23-24']
summary_table_data = summary_data.round(1).values.tolist()
summary_table_headers = ['Sector', 'Total Cos', '2024: AA1000 %', '2024: Avg Fields', '2024: % Any', 'Trend 23-24']


# =============================================================================
# ENHANCED PDF REPORT WITH STATISTICAL ANALYSIS
# =============================================================================

def create_pdf_report():
    # Import colors locally to avoid conflicts
    from reportlab.lib import colors

    # Create PDF document
    pdf_filename = f"AA1000_Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
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
    story.append(Paragraph("AA1000 Adoption Analysis by ATECO Sector", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Analysis Date: {datetime.datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph("Dataset: Tidier_Dataset.csv (AA1000 for 2023-2024)", styles['Normal']))
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
        ['AA1000 Adoption Rate (2024)', f"{key_stats['overall_completion']:.1f}%"],
        ['Companies with AA1000 (2024)', f"{key_stats['pct_with_any']:.1f}%"],
        ['Average Fields per Company (2024)', f"{key_stats['avg_fields_per_company']:.2f}"],
        ['Median Fields per Company (2024)', f"{key_stats['median_fields']:.1f}"],
        ['Standard Deviation (2024)', f"{key_stats['std_fields']:.2f}"],
        ['Trend 2023-2024', f"{trend_stats['overall_completion_growth']:+.1f}%"],
        ['Adoption Growth 2023-2024', f"{trend_stats['pct_with_any_growth']:+.1f}%"],
        ['High Adoption Sectors (>40%)', f"{len(sector_classifications.get('high_performers', pd.DataFrame()))}"],
        ['Medium Adoption Sectors (20-40%)', f"{len(sector_classifications.get('medium_performers', pd.DataFrame()))}"],
        ['Low Adoption Sectors (<20%)', f"{len(sector_classifications.get('low_performers', pd.DataFrame()))}"]
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
    story.append(Paragraph("This section shows AA1000 adoption distribution across companies.", normal_style))

    # Distribution statistics table
    dist_headers = ['AA1000 Status', 'Companies (2023)', '%', 'Companies (2024)', '%']
    dist_data = [dist_headers]

    # Show distribution for 0 and 1
    for status, label in [(0, "Not Adopted"), (1, "Adopted")]:
        row = [label]
        for year in ['2023', '2024']:
            year_dist = advanced_stats['distribution_data'][year]
            if status < len(year_dist):
                companies = year_dist[status]['companies']
                percentage = year_dist[status]['percentage']
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

    story.append(Paragraph(f"• Most common status: {mode_str} (0=Not Adopted, 1=Adopted)", bullet_style))
    story.append(
        Paragraph(f"• 25% of companies: ≤{advanced_stats['yearly_stats']['2024']['q1_fields']:.1f}", bullet_style))
    story.append(
        Paragraph(f"• 50% of companies: ≤{advanced_stats['yearly_stats']['2024']['median_fields']:.1f} (median)",
                  bullet_style))
    story.append(
        Paragraph(f"• 75% of companies: ≤{advanced_stats['yearly_stats']['2024']['q3_fields']:.1f}", bullet_style))

    story.append(PageBreak())

    # Section 2: Absolute Value Visualizations
    story.append(Paragraph("2. Absolute Count Charts by Year", heading1_style))
    story.append(Paragraph("Each bar shows the number of companies with/without AA1000 in each sector.", normal_style))
    story.append(Paragraph("How to read: Green = AA1000 adopted, Red = Not adopted", bullet_style))
    story.append(Paragraph("Sectors sorted by number of adopters (descending)", bullet_style))
    story.append(Paragraph("n= shows total number of companies in that sector", bullet_style))

    for year, image_path in zip(['2023', '2024'], image_paths_absolute):
        if image_path and os.path.exists(image_path):
            content_to_keep_together = []
            content_to_keep_together.append(Paragraph(f"{year} - AA1000 Adoption (Absolute Counts)", heading2_style))
            try:
                content_to_keep_together.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                content_to_keep_together.append(Paragraph("Image not available", normal_style))
            content_to_keep_together.append(Spacer(1, 5))

            story.append(KeepTogether(content_to_keep_together))

    story.append(PageBreak())

    # Section 3: Percentage Visualizations
    story.append(Paragraph("3. Percentage Charts by Year", heading1_style))
    story.append(Paragraph("Each bar shows the percentage of companies with AA1000 in each sector.", normal_style))
    story.append(Paragraph("How to read: Taller bars = higher adoption rate", bullet_style))
    story.append(Paragraph("Sectors sorted by adoption percentage (descending)", bullet_style))
    story.append(Paragraph("n= shows total number of companies in that sector", bullet_style))

    for year, image_path in zip(['2023', '2024'], image_paths_percentage):
        if image_path and os.path.exists(image_path):
            content_to_keep_together = []
            content_to_keep_together.append(Paragraph(f"{year} - AA1000 Adoption Rate (%)", heading2_style))
            try:
                content_to_keep_together.append(Image(image_path, width=10 * inch, height=5 * inch))
            except:
                content_to_keep_together.append(Paragraph("Image not available", normal_style))
            content_to_keep_together.append(Spacer(1, 5))

            story.append(KeepTogether(content_to_keep_together))

    story.append(PageBreak())

    # Section 4: Sector Performance Classification
    story.append(Paragraph("4. Sector Adoption Classification", heading1_style))
    story.append(Paragraph("Note: Classification based on average adoption across 2023-2024", normal_style))
    story.append(Spacer(1, 10))

    # High Performers
    high_performers = sector_classifications.get('high_performers', pd.DataFrame())
    if len(high_performers) > 0:
        story.append(Paragraph(f"High Adoption Sectors (>40%): {len(high_performers)} sectors", heading2_style))
        story.append(Paragraph("These sectors have the highest AA1000 adoption rates:", normal_style))
        for _, row in high_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg adoption, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend (2023-2024): {row.get('trend_overall_23_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No high adoption sectors found", normal_style))

    story.append(Spacer(1, 10))

    # Medium Performers
    medium_performers = sector_classifications.get('medium_performers', pd.DataFrame())
    if len(medium_performers) > 0:
        story.append(Paragraph(f"Medium Adoption Sectors (20-40%): {len(medium_performers)} sectors", heading2_style))
        story.append(Paragraph("These sectors have moderate AA1000 adoption rates:", normal_style))
        for _, row in medium_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg adoption, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend (2023-2024): {row.get('trend_overall_23_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No medium adoption sectors found", normal_style))

    story.append(Spacer(1, 10))

    # Low Performers
    low_performers = sector_classifications.get('low_performers', pd.DataFrame())
    if len(low_performers) > 0:
        story.append(Paragraph(f"Low Adoption Sectors (<20%): {len(low_performers)} sectors", heading2_style))
        story.append(Paragraph("These sectors have the lowest AA1000 adoption rates:", normal_style))
        for _, row in low_performers.sort_values('avg_overall_completion', ascending=False).iterrows():
            story.append(Paragraph(
                f"• {row['Sector']}: {row.get('avg_overall_completion', 0):.1f}% avg adoption, "
                f"{row.get('Total_Companies', 0)} companies, "
                f"Trend (2023-2024): {row.get('trend_overall_23_24', 0):+.1f}%",
                bullet_style))
    else:
        story.append(Paragraph("No low adoption sectors found", normal_style))

    story.append(PageBreak())

    # Section 5: Performance Summary Table
    story.append(Paragraph("5. Performance Summary Table (2024)", heading1_style))
    story.append(Paragraph("All sectors sorted by AA1000 adoption percentage in 2024 (descending)", normal_style))

    # Add column explanations
    story.append(Paragraph("Column Explanations:", heading2_style))
    story.append(Paragraph("• Sector: ATECO sector letter", bullet_style))
    story.append(Paragraph("• Total Cos: Total number of companies in the sector", bullet_style))
    story.append(Paragraph("• 2024: AA1000 %: Percentage of companies with AA1000 in 2024", bullet_style))
    story.append(Paragraph("• 2024: Avg Fields: Average number of fields completed per company (0 or 1)", bullet_style))
    story.append(Paragraph("• 2024: % Any: Same as AA1000 % (always 0 or 1)", bullet_style))
    story.append(
        Paragraph("• Trend 23-24: Change in adoption from 2023 to 2024 (positive = improvement)", bullet_style))
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

    # Section 6: Overall Adoption Trends
    story.append(Paragraph("6. Overall Adoption Trends (All Sectors Combined)", heading1_style))
    story.append(Paragraph("This table shows how AA1000 adoption has changed from 2023 to 2024:", normal_style))

    overall_data = [
        ['Year', 'AA1000 Adoption %', 'Companies with AA1000 %', 'Avg Fields/Company', 'Trend'],
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
            f"✓ AA1000 adoption increased by {trend_stats['overall_completion_growth']:.1f}% from 2023 to 2024",
            bullet_style))
    else:
        story.append(Paragraph(
            f"✗ AA1000 adoption decreased by {abs(trend_stats['overall_completion_growth']):.1f}% from 2023 to 2024",
            bullet_style))

    if trend_stats['pct_with_any_growth'] > 0:
        story.append(
            Paragraph(f"✓ More companies now have AA1000 (+{trend_stats['pct_with_any_growth']:.1f}%)",
                      bullet_style))
    else:
        story.append(Paragraph(f"✗ Fewer companies now have AA1000 (-{abs(trend_stats['pct_with_any_growth']):.1f}%)",
                               bullet_style))

    story.append(PageBreak())

    # Section 7: Key Findings and Insights
    story.append(Paragraph("7. Key Findings and Insights", heading1_style))

    insights = [
        f"1. Overall Adoption: In 2024, {key_stats['overall_completion']:.1f}% of companies adopted AA1000.",
        f"2. Company Participation: {key_stats['pct_with_any']:.1f}% of companies have AA1000 in 2024.",
        f"3. Average Adoption: Each company has an average of {key_stats['avg_fields_per_company']:.1f} fields (always 0 or 1).",
        f"4. Most Common: The most common status is {mode_str} (0=Not Adopted, 1=Adopted).",
        f"5. Yearly Growth: From 2023 to 2024, adoption changed by {trend_stats['overall_completion_growth']:+.1f}%.",
        f"6. High Adoption Sectors: {len(sector_classifications.get('high_performers', pd.DataFrame()))} sectors have >40% adoption.",
        f"7. Low Adoption Sectors: {len(sector_classifications.get('low_performers', pd.DataFrame()))} sectors have <20% adoption."
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

print(f"\n✓ Statistical Summary:")
if 'yearly_stats' in advanced_stats and '2024' in advanced_stats['yearly_stats']:
    key_stats = advanced_stats['yearly_stats']['2024']
    print(f"  - AA1000 adoption (2024): {key_stats['overall_completion']:.1f}%")
    print(f"  - Companies with AA1000 (2024): {key_stats['pct_with_any']:.1f}%")
    print(f"  - Average fields per company (2024): {key_stats['avg_fields_per_company']:.2f}")
    print(f"  - Standard deviation (2024): {key_stats['std_fields']:.2f}")

if 'trend_stats' in advanced_stats:
    print(f"  - Adoption trend 2023-2024: {advanced_stats['trend_stats']['overall_completion_growth']:+.1f}%")

print(f"  - High adoption sectors: {len(sector_classifications.get('high_performers', pd.DataFrame()))}")
print(f"  - Medium adoption sectors: {len(sector_classifications.get('medium_performers', pd.DataFrame()))}")
print(f"  - Low adoption sectors: {len(sector_classifications.get('low_performers', pd.DataFrame()))}")

print(f"\n✓ Image files created:")
valid_abs = [p for p in image_paths_absolute if p is not None]
valid_pct = [p for p in image_paths_percentage if p is not None]
print(f"  - Absolute charts: {len(valid_abs)} created")
print(f"  - Percentage charts: {len(valid_pct)} created")