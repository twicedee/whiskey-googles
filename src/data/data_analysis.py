import pandas as pd

def generate_report(df):
    """Generate comprehensive data quality report"""
    report = []
    
    # 1. Dataset Overview
    report.append("=== DATASET OVERVIEW ===")
    report.append(f"Total entries: {len(df)}")
    report.append(f"Columns: {list(df.columns)}\n")
    
    # 2. Missing Values Analysis
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_report = pd.DataFrame({
        "Missing Values": missing_data,
        "Percentage (%)": missing_percent.round(2)
    })
    
    report.append("=== MISSING VALUES ===")
    report.append(missing_report.to_string())
    
    # 3. Incomplete Bottles (Missing 2+ columns)
    missing_counts = df.isnull().sum(axis=1)
    incomplete_bottles = df[missing_counts >= 2]
    complete_bottles = df[missing_counts < 2]
    
    report.append("\n=== INCOMPLETE BOTTLES (MISSING 2+ COLUMNS) ===")
    report.append(f"Count: {len(incomplete_bottles)} ({len(incomplete_bottles)/len(df)*100:.1f}%)")
    report.append("\nMost common missing columns in incomplete bottles:")
    report.append(incomplete_bottles.isnull().sum().sort_values(ascending=False).to_string())
    
    # 4. Critical Missing Rows
    critical_columns = ["name", "brand_id", "image_url", "proof", "abv"]
    missing_rows = df[df[critical_columns].isnull().any(axis=1)]
    report.append("\n=== ROWS WITH MISSING CRITICAL DATA ===")
    report.append(f"Count: {len(missing_rows)}")
    report.append(missing_rows[["id", "name"] + critical_columns].to_string())
    
    # 5. Duplicates
    report.append("\n=== DUPLICATE ROWS (Name + Brand) ===")
    duplicates = df[df.duplicated(subset=["name", "brand_id"], keep=False)]
    report.append(f"Count: {len(duplicates)}")
    report.append(duplicates[["id", "name", "brand_id"]].to_string())
    
    # 6. Data Types
    report.append("\n=== DATA TYPES ===")
    report.append(df.dtypes.to_string())
    
    return "\n".join(report), complete_bottles, incomplete_bottles

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("dataset/processed/clean_image_whisky_dataset.csv")  # Update path as needed
    
    # Generate report and get separated data
    report, complete_bottles, incomplete_bottles = generate_report(df)
    
    # Save report
    with open("dataset/reports/data_quality_report.txt", "w") as f:
        f.write(report)
    
    # Save cleaned datasets
    complete_bottles.to_csv("dataset/processed/complete_bottles.csv", index=False)
    incomplete_bottles.to_csv("dataset/processed/incomplete_bottles.csv", index=False)
    
    print("Report generated and datasets separated successfully!")