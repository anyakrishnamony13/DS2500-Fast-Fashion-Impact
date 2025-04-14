'''
DS 2500
Final Project: Environmental Impact of Fast Fashion Brands
4/1/25
'''

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

FILENAME = 'Sustainable Fashion Export 2025-04-06 19-58-02.csv'

def load_and_clean_data():

    fashion_df = pd.read_csv(FILENAME)
    fashion_df.columns = fashion_df.columns.str.strip()
    
    numeric_cols = [
        "Carbon_Footprint_MT", "Water_Usage_Liters",
        "Waste_Production_KG", "Average_Price_USD", "Product_Lines"
    ]
    
    for col in numeric_cols:
        fashion_df[col] = pd.to_numeric(fashion_df[col], errors='coerce')
    
    fashion_df['Year'] = pd.to_numeric(fashion_df['Year'], errors='coerce')
    
    fashion_df = fashion_df.dropna(subset=["Sustainability_Rating"] + numeric_cols)
    

    
    return fashion_df

def country_sustainability_analysis(df):
   
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    

    country_metrics = df.groupby('Country')[metrics].mean().reset_index()
    

    top_countries = df['Country'].value_counts().head(10).index.tolist()
    country_metrics = country_metrics[country_metrics['Country'].isin(top_countries)]
    

    for metric in metrics:
        min_val = country_metrics[metric].min()
        max_val = country_metrics[metric].max()
        country_metrics[f"{metric}_Normalized"] = (country_metrics[metric] - min_val) / (max_val - min_val)
    
    # Calculate average of normalized values (lower is better)
    country_metrics['Environmental_Impact_Score'] = country_metrics[[f"{m}_Normalized" for m in metrics]].mean(axis=1)
    
    # Sort by environmental impact score (ascending for better visualization - lower impact at top)
    country_metrics = country_metrics.sort_values('Environmental_Impact_Score')
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Bar chart for each metric
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        
        # Create horizontal bar chart
        bars = axes[row, col].barh(country_metrics['Country'], country_metrics[metric], 
                              color=plt.cm.viridis(country_metrics['Environmental_Impact_Score']))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.02
            axes[row, col].text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                           va='center', fontsize=9)
        
        axes[row, col].set_title(f'Average {metric.replace("_", " ")} by Country', fontsize=12)
        axes[row, col].grid(axis='x', linestyle='--', alpha=0.7)
    
    # Overall Environmental Impact Score (lower is better)
    bars = axes[1, 1].barh(country_metrics['Country'], country_metrics['Environmental_Impact_Score'], 
                      color=plt.cm.viridis(country_metrics['Environmental_Impact_Score']))
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.02
        axes[1, 1].text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                   va='center', fontsize=9)
    
    axes[1, 1].set_title('Overall Environmental Impact Score\n(Lower is Better)', fontsize=12)
    axes[1, 1].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('country_sustainability.png', dpi=300, bbox_inches='tight')
    plt.close()


def material_type_impact_analysis(df):
    """Analyze environmental impact by material type with better visualizations"""
    # Metrics to analyze
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    # Get material impact data
    material_impact = df.groupby('Material_Type')[metrics].mean()
    
    # Sort materials by number of brands using them (to get the most common materials)
    material_counts = df['Material_Type'].value_counts()
    top_materials = material_counts.head(8).index.tolist()
    
    # Filter to top materials
    material_impact = material_impact.loc[top_materials]
    
    # Create horizontal bar charts for each metric
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Define a consistent color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_materials)))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Sort materials by this specific metric for each chart
        sorted_data = material_impact.sort_values(by=metric)
        
        # Create horizontal bar chart
        bars = axes[i].barh(sorted_data.index, sorted_data[metric], color=colors)
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width * 1.02, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                       va='center', fontsize=10, fontweight='bold')
        
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel(title, fontsize=12)
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add ranking labels (1st, 2nd, etc.) - lower is better
        for j, (material, value) in enumerate(sorted_data.iterrows()):
            axes[i].text(-0.05 * axes[i].get_xlim()[1], bars[j].get_y() + bars[j].get_height()/2, 
                         f"#{j+1}", ha='right', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('material_impact_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a grouped bar chart for all metrics together
    plt.figure(figsize=(15, 8))
    
    # Prepare data for grouped bar chart
    materials = material_impact.index
    x = np.arange(len(materials))
    width = 0.25  # width of the bars
    
    # Scale metrics to make them comparable on the same chart
    scaled_impact = material_impact.copy()
    for metric in metrics:
        max_val = scaled_impact[metric].max()
        scaled_impact[metric] = scaled_impact[metric] / max_val
    
    # Create bars
    plt.bar(x - width, scaled_impact[metrics[0]], width, label=titles[0], color='#1f77b4')
    plt.bar(x, scaled_impact[metrics[1]], width, label=titles[1], color='#ff7f0e')
    plt.bar(x + width, scaled_impact[metrics[2]], width, label=titles[2], color='#2ca02c')
    
    # Customize chart
    plt.xlabel('Material Type', fontsize=12)
    plt.ylabel('Normalized Environmental Impact\n(Lower is Better)', fontsize=12)
    plt.title('Comparative Environmental Impact by Material Type', fontsize=16)
    plt.xticks(x, materials, rotation=45, ha='right')
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig('material_impact_grouped.png', dpi=300, bbox_inches='tight')
    plt.close()
    


def sustainability_rating_validation(df):
    """Validate if sustainability ratings correspond to actual environmental performance"""
    # Metrics to analyze
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    

    rating_performance = df.groupby('Sustainability_Rating')[metrics].mean().reset_index()
    
    # Define the order of ratings
    rating_order = ['A', 'B', 'C', 'D']
    rating_performance['Sustainability_Rating'] = pd.Categorical(
        rating_performance['Sustainability_Rating'],
        categories=rating_order,
        ordered=True
    )
    rating_performance = rating_performance.sort_values('Sustainability_Rating')
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors - green for A to red for D
    colors = ['#2ca02c', '#7fbc41', '#f59a35', '#d62728']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Create bar chart
        bars = axes[i].bar(
            rating_performance['Sustainability_Rating'],
            rating_performance[metric],
            color=colors,
            width=0.7
        )
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width()/2.,
                height * 1.02,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        # Add percentage difference between A and D ratings
        a_value = rating_performance[rating_performance['Sustainability_Rating'] == 'A'][metric].values[0]
        d_value = rating_performance[rating_performance['Sustainability_Rating'] == 'D'][metric].values[0]
        pct_diff = ((d_value - a_value) / d_value) * 100
        
        axes[i].text(
            0.5, 0.9,
            f'A rating is {pct_diff:.1f}% better than D',
            transform=axes[i].transAxes,
            ha='center',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
        
        axes[i].set_title(f'Average {title} by Sustainability Rating', fontsize=14)
        axes[i].set_xlabel('Sustainability Rating', fontsize=12)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('sustainability_rating_validation.png', dpi=300, bbox_inches='tight')
    plt.close()


def market_trend_correlation(df):
    """Analyze correlation between market trends and sustainability practices"""
  
    
    # 
    if df['Market_Trend'].dtype == 'object':
        print("Converting Market_Trend to numeric")
        # Try to map string values like 'Rising', 'Stable', 'Declining' to numbers
        trend_map = {'Rising': 3, 'Stable': 2, 'Declining': 1}
        if all(trend in trend_map for trend in df['Market_Trend'].unique()):
            df['Market_Trend_Numeric'] = df['Market_Trend'].map(trend_map)
        else:
            # Just use categorical values
            df['Market_Trend_Numeric'] = pd.Categorical(df['Market_Trend']).codes
    else:
        # If it's already numeric, use as is
        df['Market_Trend_Numeric'] = df['Market_Trend']
    
    # Metrics to analyze
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    
    # Group by Market Trend
    grouped_data = df.groupby('Market_Trend_Numeric').agg({
        metric: 'mean' for metric in metrics
    }).reset_index()
    
    # Also get sustainability rating by market trend
    # Convert ratings to numeric
    rating_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
    df['Rating_Numeric'] = df['Sustainability_Rating'].map(rating_map)
    
    grouped_ratings = df.groupby('Market_Trend_Numeric').agg({
        'Rating_Numeric': 'mean'
    }).reset_index()
    
    # Merge the data
    grouped_data = pd.merge(grouped_data, grouped_ratings, on='Market_Trend_Numeric')
    
    # Sort by Market Trend
    grouped_data = grouped_data.sort_values('Market_Trend_Numeric')
    
    # Create a visualization
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Get trend labels
    if 'Market_Trend' in df.columns and df['Market_Trend'].dtype == 'object':
        unique_trends = df['Market_Trend'].unique()
        unique_numeric = df.groupby('Market_Trend')['Market_Trend_Numeric'].first().to_dict()
        trend_labels = {v: k for k, v in unique_numeric.items()}
        x_labels = [trend_labels.get(x, str(x)) for x in grouped_data['Market_Trend_Numeric']]
    else:
        
        x_labels = [str(x) for x in grouped_data['Market_Trend_Numeric']]
    
    # Plot environmental metrics as grouped bars
    bar_width = 0.2
    x = np.arange(len(grouped_data))
    
    # Normalize metrics to make them comparable
    for metric in metrics:
        grouped_data[f'{metric}_Norm'] = grouped_data[metric] / grouped_data[metric].max()
    
    # Create bars for normalized environmental metrics
    ax1.bar(x - bar_width, grouped_data[f'{metrics[0]}_Norm'], bar_width, 
           label=f'Normalized {metrics[0]}', color='#1f77b4')
    ax1.bar(x, grouped_data[f'{metrics[1]}_Norm'], bar_width, 
           label=f'Normalized {metrics[1]}', color='#ff7f0e')
    ax1.bar(x + bar_width, grouped_data[f'{metrics[2]}_Norm'], bar_width, 
           label=f'Normalized {metrics[2]}', color='#2ca02c')
    
    # Configure first y-axis (environmental metrics)
    ax1.set_xlabel('Market Trend', fontsize=12)
    ax1.set_ylabel('Normalized Environmental Impact\n(Lower is Better)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create second y-axis for sustainability rating
    ax2 = ax1.twinx()
    ax2.plot(x, grouped_data['Rating_Numeric'], 'r-', marker='o', linewidth=2, 
            label='Avg. Sustainability Rating')
    ax2.set_ylabel('Sustainability Rating\n(Higher is Better)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0.5, 4.5)  # Ratings go from 1 to 4
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_yticklabels(['D', 'C', 'B', 'A'])
    
    # Add legend for second y-axis
    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines, labels, loc='upper right')
    
    plt.title('Environmental Impact & Sustainability Rating by Market Trend', fontsize=16)
    plt.tight_layout()
    plt.savefig('market_trend_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()


def environmental_impact_over_time(df):
    """Analyze changes in environmental impact metrics over time"""
    # Check if we have multiple years
    year_counts = df['Year'].value_counts()
    valid_years = sorted(year_counts[year_counts > 5].index.tolist())
    
    if len(valid_years) <= 1:
        print("Not enough years with sufficient data for time analysis")
        return
    
    # Metrics to analyze
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Calculate yearly averages
        yearly_avg = df.groupby('Year')[metric].mean().reset_index()
        yearly_avg = yearly_avg.sort_values('Year')
        
        # Create line plot
        axes[i].plot(yearly_avg['Year'], yearly_avg[metric], 
                    marker='o', linestyle='-', linewidth=2, color='blue',
                    label='Overall Average')
        
        # Add data labels
        for x, y in zip(yearly_avg['Year'], yearly_avg[metric]):
            axes[i].text(x, y*1.02, f'{y:.1f}', ha='center', fontsize=9)
        
        # Also show by sustainability rating
        for rating, color in zip(['A', 'B', 'C', 'D'], ['green', 'lightgreen', 'orange', 'red']):
            rating_data = df[df['Sustainability_Rating'] == rating]
            if len(rating_data) > 0:
                rating_yearly = rating_data.groupby('Year')[metric].mean().reset_index()
                if len(rating_yearly) > 1:  # Only plot if we have multiple years
                    rating_yearly = rating_yearly.sort_values('Year')
                    axes[i].plot(rating_yearly['Year'], rating_yearly[metric],
                               marker='s', linestyle='--', linewidth=1.5, color=color,
                               label=f'Rating {rating}')
        
        # Calculate and show percentage change from first to last year
        if len(yearly_avg) >= 2:
            first_val = yearly_avg.iloc[0][metric]
            last_val = yearly_avg.iloc[-1][metric]
            pct_change = ((last_val - first_val) / first_val) * 100
            
            change_text = f"Change: {pct_change:.1f}% from {yearly_avg.iloc[0]['Year']} to {yearly_avg.iloc[-1]['Year']}"
            change_color = 'green' if pct_change < 0 else 'red'
            
            axes[i].text(0.5, 0.05, change_text,
                       transform=axes[i].transAxes, ha='center',
                       fontsize=12, fontweight='bold', color=change_color,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        
        # Add title and labels
        axes[i].set_title(f'{title} Over Time', fontsize=14)
        axes[i].set_xlabel('Year', fontsize=12)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('environmental_impact_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
  

def material_impact_over_time(df):
    """Analyze how material environmental impact has changed over time"""
    # Check if we have multiple years
    year_counts = df['Year'].value_counts()
    valid_years = sorted(year_counts[year_counts > 5].index.tolist())
    
    if len(valid_years) <= 1:
        print("Not enough years with sufficient data for material time analysis")
        return
    
    # Get most common materials
    material_counts = df['Material_Type'].value_counts()
    top_materials = material_counts.head(4).index.tolist()  # Top 4 materials
    
    # Filter for top materials and valid years
    df_filtered = df[(df['Material_Type'].isin(top_materials)) & (df['Year'].isin(valid_years))]
    
    # Metrics to analyze
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    # Create a figure for each metric
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.figure(figsize=(12, 7))
        
        # Plot a line for each material
        for material, color in zip(top_materials, plt.cm.tab10(np.linspace(0, 1, len(top_materials)))):
            material_data = df_filtered[df_filtered['Material_Type'] == material]
            yearly_avg = material_data.groupby('Year')[metric].mean().reset_index()
            yearly_avg = yearly_avg.sort_values('Year')
            
            if len(yearly_avg) > 1:  # Only plot if we have multiple years
                plt.plot(yearly_avg['Year'], yearly_avg[metric],
                        marker='o', linestyle='-', linewidth=2, color=color,
                        label=material)
                
                # Calculate percentage change for each material
                first_val = yearly_avg.iloc[0][metric]
                last_val = yearly_avg.iloc[-1][metric]
                pct_change = ((last_val - first_val) / first_val) * 100
                
                # Add label at the end of the line
                plt.annotate(
                    f"{pct_change:.1f}%",
                    xy=(yearly_avg.iloc[-1]['Year'], yearly_avg.iloc[-1][metric]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va='center',
                    fontweight='bold',
                    color=color
                )
        
        # Add title and labels
        plt.title(f'{title} by Material Type Over Time', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Material Type', loc='best')
        
        plt.tight_layout()
        plt.savefig(f'material_impact_time_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    

def main():
    
    
    # Load and clean the data
    df = load_and_clean_data()
    
    # Run analysis functions
    country_sustainability_analysis(df)
    material_type_impact_analysis(df)
    sustainability_rating_validation(df)
    market_trend_correlation(df)
    
    # Add the time analysis functions
    environmental_impact_over_time(df)
    material_impact_over_time(df)
    
    print("All visualizations complete!")

if __name__ == "__main__":
    main()
