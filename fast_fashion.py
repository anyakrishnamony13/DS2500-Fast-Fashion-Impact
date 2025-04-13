'''
DS 2500
Final Project: Environmental Impact of Fast Fashion Brands
4/1/25
'''

import matplotlib
matplotlib.use('TkAgg') 

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

def boxplot_by_material(df, metric, ylabel):
    df.boxplot(column=metric, by="Material_Type", rot=90, figsize=(12, 6))
    plt.title(f"{metric} by Material Type")
    plt.suptitle("")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def pie_chart_materials(df):
    df['Material_Type'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8), startangle=90)
    plt.title("Distribution of Material Types")
    plt.ylabel("")
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
    print("\n--- K-Means Clustering (First 5 Brands) ---")
    print(df[["Brand_Name", "Cluster"]].head())

def main():
    cleaned_df = clean_data(fashion_df)
    print("\n--- Grouped Environmental Metrics by Sustainability Rating ---")
    print(cleaned_df.groupby("Sustainability_Rating")[["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]].mean())

    plot_by_rating(cleaned_df)
    boxplot_by_material(cleaned_df, "Carbon_Footprint_MT", "Metric Tons")
    boxplot_by_material(cleaned_df, "Water_Usage_Liters", "Liters")
    boxplot_by_material(cleaned_df, "Waste_Production_KG", "Kilograms")
    pie_chart_materials(cleaned_df)
    linear_regression(cleaned_df)
    kmeans_clustering(cleaned_df)

if __name__ == "__main__":
    main()




