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
from sklearn.linear_model import LinearRegression

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

def simplified_price_vs_environment(df):
    # Create a more simplified visualization
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    # Color palette for sustainability ratings
    colors = {"A": "#1f77b4", "B": "#2ca02c", "C": "#ff7f0e", "D": "#d62728"}
    
    # Create a figure with three subplots - one for each environmental metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Group by rating and calculate means
        grouped_data = df.groupby('Sustainability_Rating').agg({
            'Average_Price_USD': 'mean',
            metric: 'mean'
        }).reset_index()
        
        # Sort by sustainability rating
        rating_order = ['A', 'B', 'C', 'D']
        grouped_data['Sustainability_Rating'] = pd.Categorical(
            grouped_data['Sustainability_Rating'], 
            categories=rating_order, 
            ordered=True
        )
        grouped_data = grouped_data.sort_values('Sustainability_Rating')
        
        # Create the bar chart
        bars = axes[i].bar(
            grouped_data['Sustainability_Rating'], 
            grouped_data[metric],
            color=[colors[rating] for rating in grouped_data['Sustainability_Rating']],
            width=0.6
        )
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width()/2.,
                height * 1.05,
                f'{height:.1f}',
                ha='center', 
                va='bottom', 
                fontsize=10,
                fontweight='bold'
            )
        
        # Add pricing information under each bar
        for j, (idx, row) in enumerate(grouped_data.iterrows()):
            axes[i].text(
                j, 
                -0.05 * axes[i].get_ylim()[1],
                f'${row["Average_Price_USD"]:.0f}',
                ha='center', 
                fontsize=9
            )
        
        # Set titles and labels
        axes[i].set_title(f"{title} by Rating", fontsize=14)
        axes[i].set_xlabel("Sustainability Rating", fontsize=12)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add a text label for price
        axes[i].text(
            0.5, 
            -0.15 * axes[i].get_ylim()[1],
            "Average Price (USD)",
            ha='center',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig('simplified_environmental_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Simplified environmental impact visualization saved")

def main():
    print("Starting simplified fashion data analysis...")
    
    # Load and clean the data
    df = load_and_clean_data()
    
    # Run the simplified visualization
    simplified_price_vs_environment(df)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()

