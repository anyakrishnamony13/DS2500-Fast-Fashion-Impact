'''
DS 2500
Final Project: Environmental Impact of Fast Fashion Brands
4/1/25
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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
    
    print(f"Data loaded with {len(fashion_df)} rows")
    return fashion_df

def create_scatter_plots(df):
    # Create figure for environmental metrics
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Metrics to compare
    x_metric = "Average_Price_USD"
    y_metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    y_titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    # Create scatter plots with regression lines
    for i, (y_metric, y_title) in enumerate(zip(y_metrics, y_titles)):
        # Create a basic scatter plot
        sns.scatterplot(
            x=x_metric,
            y=y_metric,
            hue="Sustainability_Rating",  # Color by rating
            size="Product_Lines",         # Size by product lines
            sizes=(20, 200),              # Range of point sizes
            alpha=0.7,                    # Transparency
            palette="viridis",            # Color palette
            data=df,
            ax=axes[i]
        )
        
        # Add regression line
        sns.regplot(
            x=x_metric,
            y=y_metric,
            data=df,
            scatter=False,  # Don't plot the points again
            ax=axes[i],
            line_kws={"color": "red", "lw": 2}
        )
        
        # Add title and labels
        axes[i].set_title(f"{y_title} vs Price", fontsize=14)
        axes[i].set_xlabel("Average Price (USD)", fontsize=12)
        axes[i].set_ylabel(y_title, fontsize=12)
        
        # Add grid lines
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Calculate correlation and add as text
        corr = df[[x_metric, y_metric]].corr().iloc[0, 1]
        axes[i].text(
            0.05, 0.95,
            f"Correlation: {corr:.2f}",
            transform=axes[i].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig('price_vs_environmental_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create scatter plot matrix for all environmental metrics
    environmental_metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    
    plt.figure(figsize=(12, 10))
    scatter_matrix = sns.pairplot(
        df[environmental_metrics + ["Sustainability_Rating", "Material_Type"]],
        hue="Sustainability_Rating",
        diag_kind="kde",  # Kernel density on diagonal
        plot_kws={"alpha": 0.6, "s": 80},
        height=2.5
    )
    
    scatter_matrix.fig.suptitle('Relationships Between Environmental Metrics', y=1.02, fontsize=16)
    plt.savefig('environmental_metrics_scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a scatter plot for price vs product lines colored by environmental impact
    plt.figure(figsize=(12, 8))
    
    # Create an environmental impact score (lower is better)
    df['Env_Impact_Score'] = 0
    for metric in environmental_metrics:
        # Normalize each metric between 0 and 1
        min_val = df[metric].min()
        max_val = df[metric].max()
        df[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val)
        df['Env_Impact_Score'] += df[f'{metric}_norm']
    
    # Create scatter plot
    scatter = plt.scatter(
        df["Average_Price_USD"],
        df["Product_Lines"],
        c=df["Env_Impact_Score"],
        cmap="RdYlGn_r",  # Red for high impact, green for low impact
        s=100,
        alpha=0.7
    )
    
    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Environmental Impact Score (Lower is Better)', rotation=270, labelpad=20)
    
    # Add title and labels
    plt.title('Price vs Product Lines by Environmental Impact', fontsize=16)
    plt.xlabel('Average Price (USD)', fontsize=14)
    plt.ylabel('Number of Product Lines', fontsize=14)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('price_productlines_envimpact_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Scatter plot analysis complete!")

def main():
    df = load_and_clean_data()
    create_scatter_plots(df)

if __name__ == "__main__":
    main()
