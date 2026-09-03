import pandas as pd

for name, path in [("Lok Sabha", "data/raw/Lok_sabha.csv"), ("Rajya Sabha", "data/raw/Rajya_sabha.csv")]:
    df = pd.read_csv(path)
    print(f"\n=== {name} ===")
    print(df.columns.tolist())
    print(df.tail(3).to_string(index=False))

    amt = pd.to_numeric(
        df["Allocated AMOUNT ( ₹ )"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )
    print("max_alloc:", amt.max())

    grand_total_count = df["Hon'ble Members of Parliaments"].astype(str).str.lower().str.contains("grand total").sum()
    print("grand_total_rows:", grand_total_count)
    print("rows:", len(df))