# Python script to demonstrate Supermarket Cutomer segmentation using DBSCAN
# Features used :
# Annual spending in Kes.
# Visit frequency (visits/per year)
# Average basket value in Kes. Customer Tenure in months
# Loyalty Score (0 - 100)

# ============================================================================
# 0. Import required modules
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. Data generation (1000 Customers with realistic segments)
# ============================================================================
def generate_customers(n_customers: int = 1000, seed: int = 42 ) -> pd.DataFrame:
    """
    Generates a synthetic dataset of customers with embedded behaviour patterns.

    Segments embedded:
    - Budget Regulars
    - Premium Loyalists
    - Bulk Buyers
    - Young / New Explorers
    - Long-term steady customers
    Args:
         n_customers(int, optional): Number of customers to generate. Defaults to 1000.):
         seed(int): For reproducibility. Defaults to 42.

    Returns:
         Dataframe with 6 Behavioural features.

    """
    rng = np.random.default_rng(seed)
    records = []
    segments = [
        ("Budget Regulars", 0.28, 45000, 8000, 96, 12, 470, 60, 36, 10, 62, 8, 4.2, 0.8),
        ("Premium Loyalists", 0.20, 280000, 45000, 78, 10, 3600, 400, 60, 15, 88, 6, 8.5, 1.0),
        ("Bulk Buyers", 0.18, 120000, 30000, 18, 6, 6500, 800, 24, 12, 55, 12, 5.0, 1.2),
        ("New Explorers", 0.17, 60000, 15000, 52, 14, 1150, 200, 9, 6, 45, 10, 9.2, 1.3),
        ("Loyal Long-Term", 0.12, 90000, 18000, 42, 8, 2100, 300, 72, 18, 78, 7, 3.5, 0.7),
    ]

    for name, pct, sp_m, sp_s, fr_m, fr_s, bk_m, bk_s, tn_m, tn_s, ly_m, ly_s, ct_m, ct_s in segments:
        n = int(n_customers * pct)

        for _ in range(n):
            records.append({
                "annual_spend_kes": np.clip(rng.normal(sp_m, sp_s), 5000, 600000),
                "visit_frequency": int(np.clip(rng.normal(fr_m, fr_s), 4, 150)),
                "avg_basket_kes": np.clip(rng.normal(bk_m, bk_s), 200, 15000),
                "tenure_months": int(np.clip(rng.normal(tn_m, tn_s), 1, 120)),
                "loyalty_score": np.clip(rng.normal(ly_m, ly_s), 0, 100),
                "category_diversity": int(np.clip(rng.normal(ct_m, ct_s), 1, 12)),
            })

    return pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)

# ============================================================================
# 2. Preprocessing
# ============================================================================
def preprocess(df: pd.DataFrame):
    """
    Standardize the features. (critical for DBSCAN distance calculations).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled

# ============================================================================
# 1. Data generation (1000 Customers with realistic segments)
# ============================================================================
def run_dbscan(X, eps = .8, min_samples = 12):
    """
    Run DBSCAN clustering
    Args:
        eps: Neighbourhod radius
        min_samples: minimum points to form a cluster
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels

# ============================================================================
# 4. Assign business-friendly segment names
# ============================================================================
def assign_business_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["cluster"] = labels
    segment_map = {}

    for cluster in set(labels):
        if cluster == -1:
            segment_map[cluster] = "Outliers"
            continue

        group = df[df["cluster"] == cluster]

        spend = group['annual_spend_kes'].mean()
        freq = group['visit_frequency'].mean()
        basket = group['avg_basket_kes'].mean()
        tenure = group['tenure_months'].mean()
        loyalty = group['loyalty_score'].mean()
        diversity = group['category_diversity'].mean()

        if spend < 80000 and freq > 80:
            name = "Budget Regulars"
        elif spend > 200000 and freq > 75:
            name = "Premium Loyalists"
        elif spend > 100000 and freq > 30 and basket > 4000:
            name = "Bulk Buyers"
        elif tenure > 60 and loyalty > 70:
            name = "Loyal Long-Term Customers"
        elif spend > 50 and diversity > 7 and tenure < 18:
            name = "New Explorers"
        else:
            name = "Mid-value Customers"

        segment_map[cluster] = name

    df["segment"] = df["cluster"].map(segment_map)

    return df

# ============================================================================
# 5. Summary Table
# ============================================================================
def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean business summary table
    Args:
    """
    total = len(df)
    rows = []

    for seg in df["segment"].unique():
        group = df[df["segment"] == seg]

        size = len(group)
        pct = size / total * 100

        spend = group['annual_spend_kes'].mean()
        freq = group['visit_frequency'].mean()
        tenure = group['tenure_months'].mean()
        basket = group['avg_basket_kes'].mean()
        diversity = group['category_diversity'].mean()
        loyalty = group['loyalty_score'].mean()

        tags = []

        if spend > 200000:
            tags.append("High Spend")
        elif spend > 80000:
            tags.append("Budget Spend")
        else:
            tags.append("Mid Spend")

        if freq > 80:
            tags.append("Very frequent")
        elif freq > 50:
            tags.append("Frequent")
        elif freq < 30:
            tags.append("Infrequent visitors")

        if basket > 4000:
            tags.append("Bulk baskets")

        if tenure > 60:
            tags.append("Long-standing")
        elif tenure < 12:
            tags.append("New Customers")

        if diversity > 7:
            tags.append("Wide category range")

        if loyalty > 75:
            tags.append("Loyal")

        rows.append({
            "Segment": seg,
            "Size": f"{size} ({pct:.1f}%)",
            "Profile": " . ".join(tags),
        })

# ============================================================================
# 6. Visualisation (live plots)
# ============================================================================
def plot_pca_clusters(X, labels):
    """PCA projection of clusters (2D visualization)"""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for cluster in set(labels):
        mask = labels == cluster
        name = "Outliers" if cluster == -1 else f"Cluster ({cluster})"

        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=.7)

    plt.title("DBSCAN Clusters (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

def plot_business_view(df):
    """Spend vs Frequency (Most intuitive business chart)"""
    plt.figure(figsize=(8, 6))

    for segment in df["segment"].unique():
        subset = df[df["segment"] == segment]

        plt.scatter(
            subset["visit_frequency"],
            subset["annual_spend_kes"],
            label=segment,
            alpha=.7
        )

    plt.xlabel("Visit Frequency")
    plt.ylabel("Annual Spend Kes")
    plt.title("Customers Segments (Business View)")
    plt.legend()
    plt.show()

# Run the application
if __name__ == "__main__":
    print("Generating dataset")
    df = generate_customers()

    print("Preprocessing")
    X = preprocess(df)

    print("Running DBSCAN")
    labels = run_dbscan(X)

    print("Assigning business segments")
    df = assign_business_segments(df, labels)

    print("\nSegment summary")
    print(segment_summary(df))

    print("\nPlotting PCA Clusters...")
    plot_pca_clusters(X, labels)

    print("Plotting business view...")
    plot_business_view(df)