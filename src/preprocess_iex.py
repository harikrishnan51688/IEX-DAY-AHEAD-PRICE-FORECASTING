
import pandas as pd
from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "DAM_Market Snapshot.xlsx"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "iex_dam_september_2025_clean.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD EXCEL DATA
# --------------------------------------------------

# header=4 means the fifth row contains column names.
df = pd.read_excel(INPUT_FILE, header=4)

# Clean column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace("*", "", regex=False)
    .str.strip()
)

print("Original columns:")
print(df.columns.tolist())

print("\nOriginal shape:", df.shape)


# --------------------------------------------------
# 3. REMOVE DAILY SUMMARY ROWS
# --------------------------------------------------

# Actual records have time blocks such as:
# 00:00 - 00:15
# 00:15 - 00:30
#
# Summary rows have values such as:
# Total (MWh), Max (MW), Min (MW), Avg (MW)

time_pattern = r"^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}$"

df = df[
    df["Time Block"]
    .astype(str)
    .str.strip()
    .str.match(time_pattern)
].copy()

print("\nShape after removing summary rows:", df.shape)


# --------------------------------------------------
# 4. CONVERT NUMERIC COLUMNS
# --------------------------------------------------

numeric_columns = [
    "Hour",
    "Purchase Bid (MW)",
    "Sell Bid (MW)",
    "MCV (MW)",
    "Final Scheduled Volume (MW)",
    "MCP (Rs/MWh)"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------
# 5. CONVERT DATE AND CREATE TIMESTAMP
# --------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Extract starting time from Time Block
df["start_time"] = (
    df["Time Block"]
    .astype(str)
    .str.extract(r"^(\d{1,2}:\d{2})")[0]
)

# Combine date and time
df["timestamp"] = pd.to_datetime(
    df["Date"].dt.strftime("%Y-%m-%d") + " " + df["start_time"],
    errors="coerce"
)


# --------------------------------------------------
# 6. HANDLE MISSING AND INVALID VALUES
# --------------------------------------------------

# Timestamp and MCP are essential for price forecasting.
# Keep rows with missing optional market variables.
df = df.dropna(subset=["timestamp", "MCP (Rs/MWh)"])

# Sort chronologically
df = df.sort_values("timestamp")

# Remove duplicate timestamps
df = df.drop_duplicates(
    subset=["timestamp"],
    keep="last"
)


# --------------------------------------------------
# 7. FEATURE ENGINEERING
# --------------------------------------------------

# Rename target variable
df["price_inr_mwh"] = df["MCP (Rs/MWh)"]

# Convert INR/MWh to INR/kWh
df["price_inr_kwh"] = df["price_inr_mwh"] / 1000

# Extract time-based features
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day_of_week"] = df["timestamp"].dt.day_of_week
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# Saturday = 5, Sunday = 6
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


# --------------------------------------------------
# 8. SELECT FINAL COLUMNS
# --------------------------------------------------

final_columns = [
    "timestamp",
    "price_inr_mwh",
    "price_inr_kwh",
    "Purchase Bid (MW)",
    "Sell Bid (MW)",
    "MCV (MW)",
    "Final Scheduled Volume (MW)",
    "hour",
    "minute",
    "day",
    "day_of_week",
    "month",
    "is_weekend"
]

df = df[final_columns]


# --------------------------------------------------
# 9. SAVE CLEANED DATASET
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# 10. DATA QUALITY REPORT
# --------------------------------------------------

print("\n========== DATA QUALITY REPORT ==========")

print("\nTotal records:", len(df))

print("Start timestamp:", df["timestamp"].min())
print("End timestamp:", df["timestamp"].max())

print("\nMissing values:")
print(df.isnull().sum())

print("\nPrice statistics:")
print(df["price_inr_mwh"].describe())

print("\nDuplicate timestamps:")
print(df["timestamp"].duplicated().sum())


# Expected timestamps for September 2025
expected = pd.date_range(
    start="2025-09-01 00:00:00",
    end="2025-09-30 23:45:00",
    freq="15min"
)

actual = pd.DatetimeIndex(df["timestamp"])

missing_intervals = expected.difference(actual)
unexpected_intervals = actual.difference(expected)

print("\nExpected September intervals:", len(expected))
print("Actual records:", len(df))
print("Missing intervals:", len(missing_intervals))
print("Unexpected intervals:", len(unexpected_intervals))

if len(missing_intervals) > 0:
    print("\nFirst missing timestamps:")
    print(missing_intervals[:10])


# --------------------------------------------------
# 11. DISPLAY CLEANED DATA
# --------------------------------------------------

print("\nFirst five records:")
print(df.head())

print("\nLast five records:")
print(df.tail())

print("\nFinal dataset information:")
print(df.info())

print("\nCleaned dataset saved to:")
print(OUTPUT_FILE)
