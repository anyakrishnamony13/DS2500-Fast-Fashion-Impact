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
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

FILENAME = 'Sustainable Fashion Export 2025-04-06 19-58-02.csv'

def load_and_clean_data():
    """Load and clean the fashion export dataset"""
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

def material_price_segment_analysis(df):
    """Analyze environmental impact by material type and price segment"""
    # Metrics to analyze
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    # Create price segments
    price_bins = [0, 100, 200, 300, 500, float('inf')]
    price_labels = ['$0-100', '$101-200', '$201-300', '$301-500', '$500+']
    df['Price_Segment'] = pd.cut(df['Average_Price_USD'], bins=price_bins, labels=price_labels)
    
    # Sort materials by number of brands using them (to get the most common materials)
    material_counts = df['Material_Type'].value_counts()
    top_materials = material_counts.head(6).index.tolist() 
    
    # Filter to top materials
    df_filtered = df[df['Material_Type'].isin(top_materials)]
    
    # Create visualizations for each metric
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.figure(figsize=(14, 8))
        
        # Calculate mean values for each material and price segment
        pivot_data = df_filtered.pivot_table(
            index='Material_Type', 
            columns='Price_Segment',
            values=metric,
            aggfunc='mean'
        ).reindex(top_materials)
        
        # Plot as a grouped bar chart
        ax = pivot_data.plot(kind='bar', figsize=(14, 8))
        
        # Calculate percentage difference from highest to lowest within each material
        for material in pivot_data.index:
            material_data = pivot_data.loc[material].dropna()
            if len(material_data) > 1:  # Only if we have multiple price points
                min_val = material_data.min()
                max_val = material_data.max()
                pct_diff = ((max_val - min_val) / max_val) * 100
                
                # Add a text annotation to show the percentage difference
                plt.annotate(
                    f'Range: {pct_diff:.1f}%',
                    xy=(top_materials.index(material), pivot_data.loc[material].max() * 1.05),
                    ha='center',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
                )
        
        plt.title(f'{title} by Material Type and Price Segment', fontsize=16)
        plt.xlabel('Material Type', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Price Segment', title_fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'material_price_segment_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Material and price segment analysis visualizations saved")

def country_material_analysis(df):
    """Analyze environmental impact by country and material type"""
    # Metrics to analyze
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    # Get top countries and materials
    top_countries = df['Country'].value_counts().head(5).index.tolist()
    top_materials = df['Material_Type'].value_counts().head(4).index.tolist()
    
    # Filter for top countries and materials
    df_filtered = df[(df['Country'].isin(top_countries)) & (df['Material_Type'].isin(top_materials))]
    
    # Create visualizations
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart using seaborn
        ax = sns.barplot(
            x='Country', 
            y=metric, 
            hue='Material_Type',
            data=df_filtered,
            palette='viridis'
        )
        
        # Add value labels on top of each bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=8)
        
        # Calculate overall average for this metric
        overall_avg = df[metric].mean()
        
        # Add a line for overall average
        plt.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7)
        plt.text(
            len(top_countries)-1, 
            overall_avg * 1.02, 
            f'Overall Avg: {overall_avg:.1f}', 
            color='red', 
            fontweight='bold'
        )
        
        plt.title(f'{title} by Country and Material Type', fontsize=16)
        plt.xlabel('Country', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Material Type', title_fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'country_material_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
 

def rating_eco_manufacturing_analysis(df):
    """Analyze environmental impact by sustainability rating and eco-friendly manufacturing"""
    # Metrics to analyze
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    # Convert Eco_Friendly_Manufacturing to categorical
    if df['Eco_Friendly_Manufacturing'].dtype == 'object':
        eco_values = {'Yes': 'Eco-Friendly', 'No': 'Not Eco-Friendly'}
        df['Eco_Category'] = df['Eco_Friendly_Manufacturing'].map(eco_values)
    else:
        df['Eco_Category'] = df['Eco_Friendly_Manufacturing'].map({1: 'Eco-Friendly', 0: 'Not Eco-Friendly'})
    
    # Set category order
    df['Sustainability_Rating'] = pd.Categorical(df['Sustainability_Rating'], 
                                              categories=['A', 'B', 'C', 'D'], 
                                              ordered=True)
    
    # Create visualizations
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart using seaborn
        ax = sns.barplot(
            x='Sustainability_Rating', 
            y=metric, 
            hue='Eco_Category',
            data=df,
            palette={'Eco-Friendly': 'green', 'Not Eco-Friendly': 'lightcoral'}
        )
        
        # Add value labels on top of each bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=10)
        
        # Add percentage difference between eco-friendly and not eco-friendly
        for rating in ['A', 'B', 'C', 'D']:
            rating_data = df[df['Sustainability_Rating'] == rating]
            eco_mean = rating_data[rating_data['Eco_Category'] == 'Eco-Friendly'][metric].mean()
            non_eco_mean = rating_data[rating_data['Eco_Category'] == 'Not Eco-Friendly'][metric].mean()
            
            if not (np.isnan(eco_mean) or np.isnan(non_eco_mean)):
                pct_diff = ((non_eco_mean - eco_mean) / non_eco_mean) * 100
                
                # Position the annotation above the bars
                rating_idx = ['A', 'B', 'C', 'D'].index(rating)
                max_val = max(eco_mean, non_eco_mean)
                
                plt.annotate(
                    f'Diff: {pct_diff:.1f}%',
                    xy=(rating_idx, max_val * 1.05),
                    ha='center',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
                )
        
        plt.title(f'{title} by Sustainability Rating and Manufacturing Type', fontsize=16)
        plt.xlabel('Sustainability Rating', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.legend(title='Manufacturing Type', title_fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'rating_eco_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    


def year_rating_trend_analysis(df):
    """Analyze trends in environmental impact over time by sustainability rating"""

    if df['Year'].nunique() <= 1:
        print("Not enough years in the dataset for trend analysis")
        return
    
    # Metrics to analyze
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    # Filter for years with substantial data
    year_counts = df['Year'].value_counts()
    valid_years = year_counts[year_counts > 5].index.tolist()
    
    if len(valid_years) <= 1:
        print("Not enough years with sufficient data for trend analysis")
        return
    
    df_filtered = df[df['Year'].isin(valid_years)]
    
    # Create visualizations
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.figure(figsize=(14, 8))
        
        # Line plot for each rating over time
        sns.lineplot(
            x='Year', 
            y=metric, 
            hue='Sustainability_Rating',
            data=df_filtered,
            palette={'A': 'green', 'B': 'lightgreen', 'C': 'orange', 'D': 'red'},
            markers=True,
            linewidth=2,
            err_style='band'
        )
        
        # Add overall trend line
        sns.lineplot(
            x='Year', 
            y=metric, 
            data=df_filtered,
            color='black',
            label='Overall',
            linestyle='--',
            linewidth=3
        )
        
        # Calculate and display percentage change
        years = sorted(valid_years)
        for rating in ['A', 'B', 'C', 'D']:
            first_year_val = df_filtered[(df_filtered['Year'] == years[0]) & 
                                       (df_filtered['Sustainability_Rating'] == rating)][metric].mean()
            last_year_val = df_filtered[(df_filtered['Year'] == years[-1]) & 
                                      (df_filtered['Sustainability_Rating'] == rating)][metric].mean()
            
            if not (np.isnan(first_year_val) or np.isnan(last_year_val)):
                pct_change = ((last_year_val - first_year_val) / first_year_val) * 100
                
                # Position at the end of each line
                plt.annotate(
                    f'{pct_change:.1f}%',
                    xy=(years[-1], last_year_val),
                    xytext=(5, 0), 
                    textcoords='offset points',
                    ha='left',
                    va='center',
                    fontweight='bold',
                    color={'A': 'green', 'B': 'darkgreen', 'C': 'darkorange', 'D': 'darkred'}[rating]
                )
        
        plt.title(f'{title} Trends by Sustainability Rating', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'year_rating_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():

    
    # Load and clean the data
    df = load_and_clean_data()
    
    # Run analysis functions
    material_price_segment_analysis(df)
    country_material_analysis(df)
    rating_eco_manufacturing_analysis(df)
    year_rating_trend_analysis(df)
    

if __name__ == "__main__":
    main()
