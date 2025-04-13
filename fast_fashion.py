'''
DS 2500
Final Project: Environmental Impact of Fast Fashion Brands
4/1/25
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Load the dataset
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

def group_by_sustainability(df):
    return df.groupby("Sustainability_Rating")[[
        "Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"
    ]].mean().sort_index()

def plot_metrics(grouped_df):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    grouped_df["Carbon_Footprint_MT"].plot(kind="bar", ax=axs[0], color='tomato')
    axs[0].set_title("Avg Carbon Footprint by Rating")
    axs[0].set_ylabel("Metric Tons")
    grouped_df["Water_Usage_Liters"].plot(kind="bar", ax=axs[1], color='skyblue')
    axs[1].set_title("Avg Water Usage by Rating")
    axs[1].set_ylabel("Liters")
    grouped_df["Waste_Production_KG"].plot(kind="bar", ax=axs[2], color='lightgreen')
    axs[2].set_title("Avg Waste Production by Rating")
    axs[2].set_ylabel("Kilograms")
    plt.tight_layout()
    plt.savefig("sustainability_plot.png")
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
    print("\n--- K-Means Clustering (First 5 Brands) ---")
    print(df[["Brand_Name", "Cluster"]].head())

def main():
    cleaned_df = clean_data(fashion_df)
    grouped = group_by_sustainability(cleaned_df)
    print("\n--- Grouped Environmental Metrics by Sustainability Rating ---")
    print(grouped)
    plot_metrics(grouped)
    linear_regression(cleaned_df)
    kmeans_clustering(cleaned_df)

if __name__ == "__main__":
    main()


