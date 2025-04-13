'''
DS 2500
Final Project: Environmental Impact of Fast Fashion Brands
4/1/25
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

FILENAME = 'Sustainable Fashion Export 2025-04-06 19-58-02.csv'
fashion_df = pd.read_csv(FILENAME)

def clean_data(df):
    df.columns = df.columns.str.strip()
    numeric_cols = [
        "Carbon_Footprint_MT", "Water_Usage_Liters",
        "Waste_Production_KG", "Average_Price_USD"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["Sustainability_Rating"] + numeric_cols)
    return df

def plot_by_rating(df):
    grouped = df.groupby("Sustainability_Rating")[[
        "Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"
    ]].mean().sort_index()
    grouped.plot(kind="bar", figsize=(12, 6), colormap="viridis")
    plt.title("Average Environmental Impact by Sustainability Rating")
    plt.ylabel("Average Value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def boxplot_by_material(df):
    df.boxplot(column="Carbon_Footprint_MT", by="Material_Type", rot=90, figsize=(12, 6))
    plt.title("Carbon Footprint by Material Type")
    plt.suptitle("")
    plt.ylabel("Metric Tons")
    plt.tight_layout()
    plt.show()

    df.boxplot(column="Water_Usage_Liters", by="Material_Type", rot=90, figsize=(12, 6))
    plt.title("Water Usage by Material Type")
    plt.suptitle("")
    plt.ylabel("Liters")
    plt.tight_layout()
    plt.show()

    df.boxplot(column="Waste_Production_KG", by="Material_Type", rot=90, figsize=(12, 6))
    plt.title("Waste Production by Material Type")
    plt.suptitle("")
    plt.ylabel("Kilograms")
    plt.tight_layout()
    plt.show()

def pie_chart_materials(df):
    df["Material_Type"].value_counts().plot.pie(
        figsize=(8, 8), autopct='%1.1f%%', startangle=90, colormap='Set3'
    )
    plt.title("Distribution of Material Types")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

def scatter_plot(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Water_Usage_Liters"], df["Waste_Production_KG"], alpha=0.6, color='purple')
    plt.title("Water Usage vs Waste Production")
    plt.xlabel("Water Usage (Liters)")
    plt.ylabel("Waste Production (KG)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def linear_regression(df):
    X = df[["Water_Usage_Liters"]]
    y = df["Waste_Production_KG"]
    model = LinearRegression()
    model.fit(X, y)
    print("\n--- Linear Regression ---")
    print(f"Coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

def kmeans_clustering(df):
    features = df[["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(features)
    print("\n--- K-Means Clustering (First 5 Rows) ---")
    print(df[["Sustainability_Rating", "Cluster"]].head())

def main():
    cleaned_df = clean_data(fashion_df)
    print("--- CLEANED DATA PREVIEW ---")
    print(cleaned_df.head())

    plot_by_rating(cleaned_df)
    boxplot_by_material(cleaned_df)
    pie_chart_materials(cleaned_df)
    scatter_plot(cleaned_df)
    linear_regression(cleaned_df)
    kmeans_clustering(cleaned_df)

if __name__ == "__main__":
    main()



