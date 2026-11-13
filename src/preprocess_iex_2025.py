
import pandas as pd
from pathlib import Path

# ==================================================
# 1. FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "iex_dam_2025_15min.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "iex_dam_2025_clean.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_FILE)

# Normalize column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
)

print("Original columns:")
print(df.columns.tolist())

print("\nOriginal shape:", df.shape)


# ==================================================
# 3. VALIDATE REQUIRED COLUMNS
# ==================================================

required_columns = [
    "date",
    "period",
    "purchase_bid",
    "sell_bid",
    "mcv",
    "final_scheduled_volume",
    "mcp",
    "congestion"
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")


# ==================================================
# 4. CLEAN TEXT COLUMNS
# ==================================================

df["date"] = df["date"].astype(str).str.strip()
df["period"] = df["period"].astype(str).str.strip()

df["congestion"] = (
    df["congestion"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# ==================================================
# 5. REMOVE INVALID PERIOD ROWS
# ==================================================

# Valid examples:
# 00:00 - 00:15
# 00:15 - 00:30

period_pattern = r"^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}$"

df = df[
    df["period"].str.match(period_pattern, na=False)
].copy()

print("\nShape after period validation:", df.shape)


# ==================================================
# 6. CONVERT NUMERIC COLUMNS
# ==================================================

numeric_columns = [
    "purchase_bid",
    "sell_bid",
    "mcv",
    "final_scheduled_volume",
    "mcp"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ==================================================
# 7. CREATE TIMESTAMP
# ==================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# Extract starting time from the period
df["start_time"] = (
    df["period"]
    .str.extract(r"^(\d{1,2}:\d{2})")[0]
)

df["timestamp"] = pd.to_datetime(
    df["date"].dt.strftime("%Y-%m-%d")
    + " "
    + df["start_time"],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(subset=["timestamp"]).copy()


# ==================================================
# 8. HANDLE DUPLICATES
# ==================================================

duplicate_count = df["timestamp"].duplicated().sum()

print("\nDuplicate timestamps found:", duplicate_count)

df = (
    df.sort_values("timestamp")
    .drop_duplicates(subset=["timestamp"], keep="last")
    .copy()
)


# ==================================================
# 9. HANDLE MISSING TARGET VALUES
# ==================================================

missing_mcp = df["mcp"].isna().sum()

print("Missing MCP values:", missing_mcp)

# MCP is our prediction target.
# Remove rows where MCP is missing.
df = df.dropna(subset=["mcp"]).copy()


# ==================================================
# 10. CONGESTION FEATURE
# ==================================================

# NO = 0, YES = 1
df["congestion"] = df["congestion"].map({
    "NO": 0,
    "YES": 1
})


# ==================================================
# 11. FEATURE ENGINEERING
# ==================================================

# Rename target
df["price_inr_mwh"] = df["mcp"]

# Convert INR/MWh to INR/kWh
df["price_inr_kwh"] = df["mcp"] / 1000

# Time-based features
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute

# Monday = 0, Sunday = 6
df["day_of_week"] = df["timestamp"].dt.day_of_week

# Weekend indicator
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ==================================================
# 12. SELECT FINAL COLUMNS
# ==================================================

final_columns = [
    "timestamp",
    "price_inr_mwh",
    "price_inr_kwh",
    "purchase_bid",
    "sell_bid",
    "mcv",
    "final_scheduled_volume",
    "congestion",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "day_of_week",
    "is_weekend"
]

df = df[final_columns]

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)


# ==================================================
# 13. DATA QUALITY REPORT
# ==================================================

print("\n========== DATA QUALITY REPORT ==========")

print("\nTotal records:", len(df))

print("\nStart timestamp:")
print(df["timestamp"].min())

print("\nEnd timestamp:")
print(df["timestamp"].max())

print("\nMissing values:")
print(df.isnull().sum())

print("\nPrice statistics:")
print(df["price_inr_mwh"].describe())

# Expected 15-minute intervals for 2025
expected = pd.date_range(
    start="2025-01-01 00:00:00",
    end="2025-12-31 23:45:00",
    freq="15min"
)

actual = pd.DatetimeIndex(df["timestamp"])

missing_intervals = expected.difference(actual)
unexpected_intervals = actual.difference(expected)

print("\nExpected intervals:", len(expected))
print("Actual records:", len(df))
print("Missing intervals:", len(missing_intervals))
print("Unexpected intervals:", len(unexpected_intervals))

if len(missing_intervals) > 0:
    print("\nFirst 20 missing timestamps:")
    print(missing_intervals[:20])

if len(unexpected_intervals) > 0:
    print("\nFirst 20 unexpected timestamps:")
    print(unexpected_intervals[:20])


# ==================================================
# 14. MONTHLY RECORD COUNT
# ==================================================

print("\n========== MONTHLY RECORD COUNT ==========")

monthly_counts = (
    df.groupby(df["timestamp"].dt.month)
    .size()
)

print(monthly_counts)


# ==================================================
# 15. SAVE CLEANED DATASET
# ==================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== FINAL DATASET ==========")

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nCleaned dataset saved to:")
print(OUTPUT_FILE)
