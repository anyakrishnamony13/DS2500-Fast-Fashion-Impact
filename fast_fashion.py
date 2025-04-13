'''
DS 2500
Final Project: Environmental Impact of Fast Fashion Brands
4/1/25
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILENAME = 'Sustainable Fashion Export 2025-04-06 19-58-02.csv'
fashion_df = pd.read_csv(FILENAME)

def clean_data(df):
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "Carbon_Footprint_MT", "Water_Usage_Liters", 
        "Waste_Production_KG", "Average_Price_USD" ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=["Sustainability_Rating"] + numeric_cols)

    return df
    
def group_by_sustainability(df):
    grouped = df.groupby("Sustainability_Rating")[
        ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    ].mean()
    return grouped

def plot_metrics(grouped_df):
    grouped_df = grouped_df.sort_index()  # Sort ratings (e.g., A to D)

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
    plt.show()
    
def main():
    cleaned = clean_data(fashion_df)
    grouped = group_by_sustainability(cleaned)
    print(grouped)  # Optional: Print average metrics
    plot_metrics(grouped)

main()
