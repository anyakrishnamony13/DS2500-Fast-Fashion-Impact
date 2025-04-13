'''
DS 2500
Final Project: Environmental Impact of Fast Fashion Brands
4/1/25
'''

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FILENAME = 'Sustainable Fashion Export 2025-04-06 19-58-02.csv'

def load_and_clean_data():
    fashion_df = pd.read_csv(FILENAME)
    
    fashion_df.columns = fashion_df.columns.str.strip()
    numeric_cols = [
        "Carbon_Footprint_MT", "Water_Usage_Liters",
        "Waste_Production_KG", "Average_Price_USD"
    ]
    for col in numeric_cols:
        fashion_df[col] = pd.to_numeric(fashion_df[col], errors='coerce')
    
    fashion_df = fashion_df.dropna(subset=["Sustainability_Rating"] + numeric_cols)
    
    return fashion_df

def plot_by_rating_improved(df):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (Metric Tons)", "Water Usage (Liters)", "Waste Production (KG)"]
    colors = ["darkblue", "teal", "gold"]
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        grouped = df.groupby("Sustainability_Rating")[[metric]].mean().sort_index()
        grouped.plot(kind="bar", ax=axes[i], color=color)
        axes[i].set_title(title)
        axes[i].set_ylabel("Average Value")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        for j, v in enumerate(grouped[metric]):
            axes[i].text(j, v * 1.03, f'{v:.1f}', ha='center', fontweight='bold')
    
    plt.xlabel("Sustainability Rating")
    plt.suptitle("Environmental Impact by Sustainability Rating", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('sustainability_rating_comparison.png')
    plt.close()

def boxplot_by_material(df, metric, ylabel):
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x="Material_Type", y=metric, data=df, palette="viridis")
    plt.title(f"{metric} by Material Type", fontsize=14)
    plt.xlabel("Material Type", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{metric}_by_material.png')
    plt.close()

def pie_chart_materials(df):
    material_counts = df['Material_Type'].value_counts()
    
    colors = plt.cm.Spectral(np.linspace(0, 1, len(material_counts)))
    explode = [0.1 if i == 0 else 0 for i in range(len(material_counts))]
    
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(
        material_counts, 
        labels=material_counts.index,
        autopct='%1.1f%%', 
        startangle=90, 
        explode=explode,
        colors=colors,
        shadow=True
    )
    
    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, size=12)
    
    plt.title("Distribution of Material Types in Fashion Brands", fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('material_distribution.png')
    plt.close()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    metrics = ['Carbon_Footprint_MT', 'Water_Usage_Liters', 
               'Waste_Production_KG', 'Average_Price_USD']
    
    corr_matrix = df[metrics].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Between Environmental Metrics and Price', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def plot_scatter_with_regression(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    x_vars = ['Water_Usage_Liters', 'Carbon_Footprint_MT', 'Average_Price_USD', 'Water_Usage_Liters']
    y_vars = ['Waste_Production_KG', 'Water_Usage_Liters', 'Carbon_Footprint_MT', 'Carbon_Footprint_MT']
    titles = [
        'Water Usage vs. Waste Production',
        'Carbon Footprint vs. Water Usage',
        'Price vs. Carbon Footprint',
        'Water Usage vs. Carbon Footprint'
    ]
    
    for i, (x_var, y_var, title) in enumerate(zip(x_vars, y_vars, titles)):
        axes[i].scatter(df[x_var], df[y_var], alpha=0.6, c='darkblue')
        
        X = df[x_var].values.reshape(-1, 1)
        y = df[y_var].values
        model = LinearRegression()
        model.fit(X, y)
        
        x_range = np.linspace(df[x_var].min(), df[x_var].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        axes[i].plot(x_range, y_pred, 'r-', linewidth=2)
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel(x_var.replace('_', ' '), fontsize=12)
        axes[i].set_ylabel(y_var.replace('_', ' '), fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        r_squared = np.corrcoef(df[x_var], df[y_var])[0, 1]**2
        eq_text = f'y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}\nR² = {r_squared:.4f}'
        axes[i].text(0.05, 0.95, eq_text, transform=axes[i].transAxes, 
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('regression_relationships.png')
    plt.close()

def kmeans_clustering_plot(df):
    features = df[["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    inertia = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('kmeans_elbow.png')
    plt.close()
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_features)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for cluster in range(3):
        cluster_data = df[df["Cluster"] == cluster]
        ax.scatter(
            cluster_data["Carbon_Footprint_MT"],
            cluster_data["Water_Usage_Liters"],
            cluster_data["Waste_Production_KG"],
            label=f'Cluster {cluster}',
            s=60,
            alpha=0.7
        )
    
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        s=200, marker='X', c='black', label='Centroids'
    )
    
    ax.set_xlabel('Carbon Footprint (MT)', fontsize=12)
    ax.set_ylabel('Water Usage (Liters)', fontsize=12)
    ax.set_zlabel('Waste Production (KG)', fontsize=12)
    ax.set_title('K-means Clustering of Fashion Brands by Environmental Impact', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('kmeans_clusters_3d.png')
    plt.close()
    
    cluster_summary = df.groupby("Cluster")[
        ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG", "Average_Price_USD"]
    ].mean()
    
    cluster_summary["Brand_Count"] = df.groupby("Cluster").size()
    
    cluster_summary["Most_Common_Rating"] = df.groupby("Cluster")["Sustainability_Rating"].agg(
        lambda x: x.value_counts().index[0]
    )
    
    return cluster_summary

def sustainability_vs_price_analysis(df):
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(x="Sustainability_Rating", y="Average_Price_USD", data=df, palette="YlGnBu")
    
    sns.stripplot(x="Sustainability_Rating", y="Average_Price_USD", data=df, 
                  size=4, color=".3", linewidth=0, alpha=0.5)
    
    plt.title("Relationship Between Sustainability Rating and Price", fontsize=14)
    plt.xlabel("Sustainability Rating", fontsize=12)
    plt.ylabel("Average Price (USD)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('sustainability_vs_price.png')
    plt.close()
    
    price_by_rating = df.groupby("Sustainability_Rating")["Average_Price_USD"].agg(['mean', 'median', 'std'])
    return price_by_rating

def main():
    print("Loading and cleaning data...")
    fashion_df = load_and_clean_data()
    
    print("\n--- Dataset Overview ---")
    print(f"Number of brands: {len(fashion_df)}")
    print(f"Number of unique materials: {fashion_df['Material_Type'].nunique()}")
    print(f"Sustainability rating distribution:\n{fashion_df['Sustainability_Rating'].value_counts()}")
    
    print("\n--- Creating visualizations... ---")
    plot_by_rating_improved(fashion_df)
    
    for metric, unit in [
        ("Carbon_Footprint_MT", "Metric Tons"), 
        ("Water_Usage_Liters", "Liters"), 
        ("Waste_Production_KG", "Kilograms")
    ]:
        boxplot_by_material(fashion_df, metric, unit)
    
    pie_chart_materials(fashion_df)
    plot_correlation_heatmap(fashion_df)
    plot_scatter_with_regression(fashion_df)
    
    print("\n--- Linear Regression Analysis ---")
    X = fashion_df[["Water_Usage_Liters"]]
    y = fashion_df["Waste_Production_KG"]
    model = LinearRegression()
    model.fit(X, y)
    print(f"Water Usage → Waste Production:")
    print(f"Coefficient: {model.coef_[0]:.6f}")
    print(f"Intercept: {model.intercept_:.4f}")
    r_squared = model.score(X, y)
    print(f"R-squared: {r_squared:.4f}")
    
    print("\n--- K-Means Clustering Analysis ---")
    cluster_summary = kmeans_clustering_plot(fashion_df)
    print("Cluster Profiles:")
    print(cluster_summary)
    
    print("\n--- Price vs Sustainability Analysis ---")
    price_analysis = sustainability_vs_price_analysis(fashion_df)
    print(price_analysis)
    
    print("\nAll analyses complete! Visualizations saved to current directory.")

if __name__ == "__main__":
    main()




