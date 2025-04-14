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
from sklearn.cluster import KMeans
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

def price_vs_environmental_metrics(df):
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.scatterplot(
            x="Average_Price_USD", 
            y=metric, 
            hue="Sustainability_Rating", 
            size="Product_Lines",
            sizes=(20, 200),
            alpha=0.7,
            data=df, 
            ax=axes[i]
        )
        
        X = df[["Average_Price_USD"]]
        y = df[metric]
        model = LinearRegression()
        model.fit(X, y)
        
        x_range = np.linspace(df["Average_Price_USD"].min(), df["Average_Price_USD"].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        axes[i].plot(x_range, y_pred, 'r-', linewidth=2)
        
        r2 = model.score(X, y)
        coef = model.coef_[0]
        
        text = f"y = {coef:.2f}x + {model.intercept_:.2f}\nR² = {r2:.2f}"
        axes[i].text(0.05, 0.95, text, transform=axes[i].transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[i].set_title(f"Price vs {title}", fontsize=14)
        axes[i].set_xlabel("Average Price (USD)", fontsize=12)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('price_vs_environmental_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
 

def environmental_impact_by_year(df):
    yearly_data = df.groupby("Year")[["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]].mean().reset_index()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Average Carbon Footprint (MT)", "Average Water Usage (Liters)", "Average Waste Production (KG)"]
    colors = ["darkblue", "teal", "darkgreen"]
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        axes[i].plot(yearly_data["Year"], yearly_data[metric], marker='o', linestyle='-', linewidth=2, color=color)
        
        for x, y in zip(yearly_data["Year"], yearly_data[metric]):
            axes[i].annotate(f"{y:.1f}", (x, y), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontweight='bold')
        
        axes[i].set_title(title, fontsize=14)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.xlabel("Year", fontsize=12)
    plt.tight_layout()
    plt.savefig('environmental_impact_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()

def material_impact_analysis(df):
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    
    plt.figure(figsize=(12, 8))
    material_impact = df.groupby('Material_Type')[metrics].mean()
    
    scaler = StandardScaler()
    material_impact_scaled = pd.DataFrame(
        scaler.fit_transform(material_impact),
        columns=material_impact.columns,
        index=material_impact.index
    )
    
    sns.heatmap(material_impact_scaled, annot=material_impact.round(1), cmap="RdYlGn_r", 
                linewidths=.5, fmt='.1f', cbar_kws={'label': 'Normalized Impact (Lower is Better)'})
    plt.title('Environmental Impact by Material Type (Normalized)', fontsize=16)
    plt.tight_layout()
    plt.savefig('material_impact_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
 

def main():


    df = load_and_clean_data()
    

    price_vs_environmental_metrics(df)
    environmental_impact_by_year(df)
    material_impact_analysis(df)
    


if __name__ == "__main__":
    main()

