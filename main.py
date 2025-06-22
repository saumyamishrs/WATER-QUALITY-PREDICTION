# main.py
from src.preprocess import load_data, clean_data

def main():
    path = "data/water_quality.csv"
    df = load_data(path)
    print("✅ Data loaded successfully.\n")
    print("🔍 Dataset Info:")
    print(df.info())
    df_clean = clean_data(df)
    print(f"\n✅ Cleaned data shape: {df_clean.shape}")
    print("\n📊 Summary statistics:")
    print(df_clean.describe())

if __name__ == "__main__":
    main()
