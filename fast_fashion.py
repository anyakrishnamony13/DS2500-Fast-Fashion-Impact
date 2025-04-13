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
    plt.show(block=True)

def boxplot_by_material(df):
    df.boxplot(column="Carbon_Footprint_MT", by="Material_Type", rot=90, figsize=(12, 6))
    plt.title("Carbon Footprint by Material Type")
    plt.suptitle("")
    plt.ylabel("Metric Tons")
    plt.tight_layout()
    plt.show(block=True)

def pie_chart_materials(df):
    counts = df["Material_Type"].value_counts()
    counts.plot.pie(autopct='%1.1f%%', figsize=(8, 8))
    plt.title("Most Common Materials Used")
    plt.ylabel("")
    plt.tight_layout()
    plt.show(block=True)

def scatter_plot(df):
    df.plot(kind='scatter', x='Water_Usage_Liters', y='Carbon_Footprint_MT',
            alpha=0.5, color='teal', figsize=(10, 6))
    plt.title("Water Usage vs. Carbon Footprint")
    plt.xlabel("Water Usage (Liters)")
    plt.ylabel("Carbon Footprint (Metric Tons)")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

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

    plot_by_rating(cleaned_df)
    input("Press Enter to see boxplot...")
    boxplot_by_material(cleaned_df)
    input("Press Enter to see pie chart...")
    pie_chart_materials(cleaned_df)
    input("Press Enter to see scatter plot...")
    scatter_plot(cleaned_df)

    linear_regression(cleaned_df)
    kmeans_clustering(cleaned_df)

if __name__ == "__main__":
    main()




