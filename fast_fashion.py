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
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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
    
    print(f"Data loaded with {len(fashion_df)} rows and {len(fashion_df.columns)} columns")
    
    return fashion_df

def sustainability_rating_violin_plots(df):
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (Metric Tons)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.violinplot(x="Sustainability_Rating", y=metric, data=df, ax=axes[i], palette="YlGnBu", inner="quartile")
        
        sns.pointplot(x="Sustainability_Rating", y=metric, data=df, ax=axes[i], 
                     estimator=np.mean, color="darkred", markers="D", scale=0.7, join=False)
        
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel("Sustainability Rating", fontsize=12)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('sustainability_rating_violinplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Sustainability rating violin plots saved")

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
    
    print("Price vs environmental metrics plots saved")

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
        
        # Calculate percent change year over year
        if len(yearly_data) > 1:
            pct_changes = yearly_data[metric].pct_change() * 100
            for j in range(1, len(yearly_data)):
                x1, x2 = yearly_data["Year"].iloc[j-1], yearly_data["Year"].iloc[j]
                y1, y2 = yearly_data[metric].iloc[j-1], yearly_data[metric].iloc[j]
                
                pct = pct_changes.iloc[j]
                color = 'green' if pct < 0 else 'red'  # Green for decrease (improvement)
                
                axes[i].annotate(f"{pct:.1f}%", ((x1+x2)/2, (y1+y2)/2), 
                               textcoords="offset points", xytext=(0,15), 
                               ha='center', color=color, fontweight='bold')
    
    plt.xlabel("Year", fontsize=12)
    plt.tight_layout()
    plt.savefig('environmental_impact_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Environmental impact by year plots saved")

def material_type_environmental_comparison(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    material_stats = df.groupby("Material_Type")[metrics].agg(['mean', 'std']).reset_index()
    
    material_types = df["Material_Type"].unique()
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        bar_width = 0.8 / len(material_types)
        positions = np.arange(len(material_types))
        
        mean_values = df.groupby("Material_Type")[metric].mean().values
        std_values = df.groupby("Material_Type")[metric].std().values
        
        bars = axes[i].bar(positions, mean_values, width=bar_width, 
                         yerr=std_values, capsize=5, 
                         color=sns.color_palette("viridis", len(material_types)),
                         alpha=0.8)
        
        for bar, value in zip(bars, mean_values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        axes[i].set_title(f"{title} by Material Type", fontsize=14)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].set_xticks(positions)
        axes[i].set_xticklabels(material_types, rotation=45, ha='right')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('material_type_barcharts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
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
    
    metrics_display = ["Carbon\nFootprint", "Water\nUsage", "Waste\nProduction"]
    
    plt.figure(figsize=(12, 10))
    
    N = len(metrics)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    material_radar = material_impact.copy()
    for metric in metrics:
        material_radar[metric] = (material_radar[metric] - material_radar[metric].min()) / \
                                (material_radar[metric].max() - material_radar[metric].min())
    
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], metrics_display, size=12)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    for i, material in enumerate(material_types):
        values = material_radar.loc[material].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=material)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Material Type Environmental Impact Comparison', size=16)
    
    plt.tight_layout()
    plt.savefig('material_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Material type environmental comparison plots saved")

def eco_friendly_manufacturing_comparison(df):
    if df['Eco_Friendly_Manufacturing'].dtype == 'object':
        eco_values = {'Yes': 1, 'No': 0}
        df['Eco_Friendly_Flag'] = df['Eco_Friendly_Manufacturing'].map(eco_values)
    else:
        df['Eco_Friendly_Flag'] = df['Eco_Friendly_Manufacturing']
    
    df['Eco_Friendly_Category'] = df['Eco_Friendly_Flag'].map({1: 'Eco-Friendly', 0: 'Not Eco-Friendly'})
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    titles = ["Carbon Footprint (MT)", "Water Usage (Liters)", "Waste Production (KG)"]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        stats = df.groupby('Eco_Friendly_Category')[metric].agg(['mean', 'std'])
        
        x_pos = np.arange(len(stats.index))
        
        bars = axes[i].bar(x_pos, stats['mean'], yerr=stats['std'], 
                         capsize=10, width=0.6, 
                         color=['lightcoral', 'lightgreen'], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=12)
        
        eco_friendly_mean = df[df['Eco_Friendly_Flag'] == 1][metric].mean()
        non_eco_friendly_mean = df[df['Eco_Friendly_Flag'] == 0][metric].mean()
        pct_diff = ((non_eco_friendly_mean - eco_friendly_mean) / non_eco_friendly_mean) * 100
        
        stat_text = f"Mean Difference: {abs(eco_friendly_mean - non_eco_friendly_mean):.1f}\n"
        stat_text += f"Percent Improvement: {pct_diff:.1f}%"
        
        axes[i].text(0.5, 0.95, stat_text, transform=axes[i].transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[i].set_title(f"{title} by Manufacturing Type", fontsize=14)
        axes[i].set_xlabel("Manufacturing Approach", fontsize=12)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(stats.index)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('eco_friendly_manufacturing_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.violinplot(x="Eco_Friendly_Category", y=metric, data=df, ax=axes[i], 
                      palette=["lightcoral", "lightgreen"], inner="quartile", split=False)
        
        sns.stripplot(x="Eco_Friendly_Category", y=metric, data=df, 
                     ax=axes[i], color=".3", alpha=0.5, jitter=True, size=3)
        
        for j, cat in enumerate(df['Eco_Friendly_Category'].unique()):
            mean_val = df[df['Eco_Friendly_Category'] == cat][metric].mean()
            axes[i].plot([j-0.2, j+0.2], [mean_val, mean_val], 'r-', linewidth=2)
            axes[i].text(j, mean_val, f'Mean: {mean_val:.1f}', ha='center', va='bottom', fontsize=10)
        
        axes[i].set_title(f"{title} by Manufacturing Type", fontsize=14)
        axes[i].set_xlabel("Manufacturing Approach", fontsize=12)
        axes[i].set_ylabel(title, fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('eco_friendly_manufacturing_violins.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Eco-friendly manufacturing comparison plots saved")

# def product_line_diversity_impact(df):

def certifications_analysis(df):
    # Assuming Certifications column contains comma-separated values
    if 'Certifications' in df.columns:
        # Create a list of all certifications
        all_certs = []
        for certs in df['Certifications'].dropna():
            if isinstance(certs, str):
                cert_list = [c.strip() for c in certs.split(',')]
                all_certs.extend(cert_list)
        
        # Count certifications
        cert_counts = pd.Series(all_certs).value_counts()
        
        # Plot certification counts - horizontal bar chart instead of vertical
        plt.figure(figsize=(12, 10))
        cert_counts.plot(kind='barh', color='teal')
        plt.title('Frequency of Certifications', fontsize=14)
        plt.ylabel('Certification', fontsize=12)
        plt.xlabel('Count', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('certification_counts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze certifications by sustainability rating
        # Create a dictionary to hold certification counts by rating
        cert_by_rating = {}
        
        for rating in df['Sustainability_Rating'].unique():
            brands_with_rating = df[df['Sustainability_Rating'] == rating]
            rating_certs = []
            
            for certs in brands_with_rating['Certifications'].dropna():
                if isinstance(certs, str):
                    cert_list = [c.strip() for c in certs.split(',')]
                    rating_certs.extend(cert_list)
            
            cert_by_rating[rating] = pd.Series(rating_certs).value_counts()
        
        # Convert to DataFrame
        cert_rating_df = pd.DataFrame(cert_by_rating).fillna(0)
        
        # Select top certifications for clarity (e.g., top 10)
        if len(cert_rating_df) > 10:
            top_certs = cert_rating_df.sum(axis=1).nlargest(10).index
            cert_rating_df = cert_rating_df.loc[top_certs]
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(cert_rating_df, annot=True, cmap="YlGnBu", fmt='g')
        plt.title('Certification Distribution by Sustainability Rating', fontsize=14)
        plt.xlabel('Sustainability Rating', fontsize=12)
        plt.ylabel('Certification', fontsize=12)
        plt.tight_layout()
        plt.savefig('certification_by_rating.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze environmental metrics by certification presence
        metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
        
        # Get most common certifications (top 5)
        top_certifications = cert_counts.nlargest(5).index
        
        # Create new columns for each top certification
        for cert in top_certifications:
            df[f'Has_{cert.replace(" ", "_")}'] = df['Certifications'].apply(
                lambda x: 1 if isinstance(x, str) and cert in x else 0
            )
        
        # Replace boxplots with bar charts for metrics by certification
        fig, axes = plt.subplots(len(metrics), len(top_certifications), figsize=(20, 15))
        
        for i, metric in enumerate(metrics):
            for j, cert in enumerate(top_certifications):
                cert_col = f'Has_{cert.replace(" ", "_")}'
                
                # Map to more readable labels
                df[cert_col] = df[cert_col].map({1: f'Has {cert}', 0: f'No {cert}'})
                
                # Calculate statistics
                cert_stats = df.groupby(cert_col)[metric].agg(['mean', 'std']).reset_index()
                
                # Create bar chart with error bars
                bars = axes[i, j].bar(cert_stats[cert_col], cert_stats['mean'], 
                                    yerr=cert_stats['std'], capsize=7,
                                    color=['grey', 'teal'], alpha=0.7)
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    axes[i, j].text(bar.get_x() + bar.get_width()/2., height + 5,
                                  f'{height:.1f}', ha='center', va='bottom')
                
                axes[i, j].set_title(f'{cert} - {metric}', fontsize=12)
                axes[i, j].set_xlabel('')
                axes[i, j].set_ylabel(metric if j == 0 else '')
                axes[i, j].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('environmental_metrics_by_certification.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Certifications analysis plots saved")
    else:
        print("No 'Certifications' column found in the dataset")

# def correlation_matrix(df):

def certifications_analysis(df):
    if 'Certifications' in df.columns:
        all_certs = []
        for certs in df['Certifications'].dropna():
            if isinstance(certs, str):
                cert_list = [c.strip() for c in certs.split(',')]
                all_certs.extend(cert_list)
        
        cert_counts = pd.Series(all_certs).value_counts()
        
        plt.figure(figsize=(12, 10))
        cert_counts.plot(kind='barh', color='teal')
        plt.title('Frequency of Certifications', fontsize=14)
        plt.ylabel('Certification', fontsize=12)
        plt.xlabel('Count', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('certification_counts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        cert_by_rating = {}
        
        for rating in df['Sustainability_Rating'].unique():
            brands_with_rating = df[df['Sustainability_Rating'] == rating]
            rating_certs = []
            
            for certs in brands_with_rating['Certifications'].dropna():
                if isinstance(certs, str):
                    cert_list = [c.strip() for c in certs.split(',')]
                    rating_certs.extend(cert_list)
            
            cert_by_rating[rating] = pd.Series(rating_certs).value_counts()
        
        cert_rating_df = pd.DataFrame(cert_by_rating).fillna(0)
        
        if len(cert_rating_df) > 10:
            top_certs = cert_rating_df.sum(axis=1).nlargest(10).index
            cert_rating_df = cert_rating_df.loc[top_certs]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cert_rating_df, annot=True, cmap="YlGnBu", fmt='g')
        plt.title('Certification Distribution by Sustainability Rating', fontsize=14)
        plt.xlabel('Sustainability Rating', fontsize=12)
        plt.ylabel('Certification', fontsize=12)
        plt.tight_layout()
        plt.savefig('certification_by_rating.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        metrics = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
        
        top_certifications = cert_counts.nlargest(5).index
        
        for cert in top_certifications:
            df[f'Has_{cert.replace(" ", "_")}'] = df['Certifications'].apply(
                lambda x: 1 if isinstance(x, str) and cert in x else 0
            )
        
        fig, axes = plt.subplots(len(metrics), len(top_certifications), figsize=(20, 15))
        
        for i, metric in enumerate(metrics):
            for j, cert in enumerate(top_certifications):
                cert_col = f'Has_{cert.replace(" ", "_")}'
                df[cert_col] = df[cert_col].map({1: f'Has {cert}', 0: f'No {cert}'})
                
                cert_stats = df.groupby(cert_col)[metric].agg(['mean', 'std']).reset_index()
                
                bars = axes[i, j].bar(cert_stats[cert_col], cert_stats['mean'], 
                                    yerr=cert_stats['std'], capsize=7,
                                    color=['grey', 'teal'], alpha=0.7)
                
                for bar in bars:
                    height = bar.get_height()
                    axes[i, j].text(bar.get_x() + bar.get_width()/2., height + 5,
                                  f'{height:.1f}', ha='center', va='bottom')
                
                axes[i, j].set_title(f'{cert} - {metric}', fontsize=12)
                axes[i, j].set_xlabel('')
                axes[i, j].set_ylabel(metric if j == 0 else '')
                axes[i, j].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('environmental_metrics_by_certification.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Certifications analysis plots saved")
    else:
        print("No 'Certifications' column found in the dataset")

# def market_trend_analysis(df):

if __name__ == "__main__":
    main()



