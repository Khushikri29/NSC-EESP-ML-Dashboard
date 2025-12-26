import pandas as pd
from sqlalchemy import create_engine

# -----------------------------
# 1. Load CSV safely
# -----------------------------
FILE_PATH = "nsc_data.csv"

try:
    df = pd.read_csv(FILE_PATH, encoding="latin1", low_memory=False)
except Exception as e:
    print("CSV load error:", e)
    exit()

# -----------------------------
# 2. Clean column names
# -----------------------------
df.columns = df.columns.str.strip()   # remove extra spaces
df.columns = df.columns.str.upper()   # make uniform (optional but useful)

print("Columns in dataset:")
print(df.columns.tolist())

# -----------------------------
# 3. Convert date columns (ONLY if present)
# -----------------------------
DATE_COLUMNS = [
    "ACK_DT",
    "ACK_DATE",
    "REQUEST_DATE",
    "CREATED_DATE"
]

for col in DATE_COLUMNS:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"Converted {col} to datetime")

# -----------------------------
# 4. Basic data check
# -----------------------------
print("\nDataset shape:", df.shape)
print("\nSample data:")
print(df.head())

# -----------------------------
# 5. Create SQLite Database
# -----------------------------
engine = create_engine("sqlite:///nsc_eesp.db")

# -----------------------------
# 6. Store data into database
# -----------------------------
df.to_sql(
    name="nsc_requests",
    con=engine,
    if_exists="replace",
    index=False
)

print("\n✅ Data successfully stored in SQLite database (nsc_eesp.db)")
