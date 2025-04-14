'''
DS 2500
Final Project: Environmental Impact of Fast Fashion Brands
4/1/25
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

def clean_and_prepare(df):
    df.columns = df.columns.str.strip()
    df["Sustainability_Score"] = df["Sustainability_Rating"].map({"A": 4, "B": 3, "C": 2, "D": 1})
    
    df["Eco_Friendly_Manufacturing"] = df["Eco_Friendly_Manufacturing"].map({"Yes": 1, "No": 0})
    df["Recycling_Programs"] = df["Recycling_Programs"].map({"Yes": 1, "No": 0})
    
    le = LabelEncoder()
    df["Material_Code"] = le.fit_transform(df["Material_Type"])
    df["Certification_Code"] = le.fit_transform(df["Certifications"].astype(str))
    
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include='number').columns] = imputer.fit_transform(df.select_dtypes(include='number'))
    
    return df


def linear_regression_analysis(df):
    X = df[["Water_Usage_Liters", "Carbon_Footprint_MT", "Eco_Friendly_Manufacturing", "Material_Code"]]
    y = df["Waste_Production_KG"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    print(f"Linear Regression R²: {model.score(X_test, y_test):.2f}")
    plt.show()

def logistic_regression_analysis(df):
    df["Is_Sustainable"] = df["Sustainability_Score"] >= 3
    X = df[["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG", "Eco_Friendly_Manufacturing"]]
    y = df["Is_Sustainable"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression().fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    plt.show()

def decision_tree_analysis(df):
    X = df[["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG", "Eco_Friendly_Manufacturing"]]
    y = df["Is_Sustainable"]
    model = DecisionTreeClassifier(max_depth=4).fit(X, y)
    plt.figure(figsize=(16, 10))
    plot_tree(model, feature_names=X.columns, class_names=["Unsustainable", "Sustainable"], filled=True)
    plt.title("Decision Tree")
    plt.show()

def kmeans_clustering(df):
    features = ["Carbon_Footprint_MT", "Water_Usage_Liters", "Waste_Production_KG"]
    X_scaled = StandardScaler().fit_transform(df[features])
    df["Cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Carbon_Footprint_MT", y="Waste_Production_KG", hue="Cluster", palette="Set2")
    plt.title("K-Means Clustering")
    plt.show()

def main():
    df = pd.read_csv("Sustainable Fashion Export 2025-04-06 19-58-02.csv")
    df = clean_and_prepare(df)
    linear_regression_analysis(df)
    logistic_regression_analysis(df)
    decision_tree_analysis(df)
    kmeans_clustering(df)


if __name__ == "__main__":
    main()



