import pandas as pd
import os

DATA_DIR = "../data/noaa_disaster_data"

event_rows = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        file_path = os.path.join(DATA_DIR, file)
        print(f"Processing: {file}")  # Debugging: Print filename

        try:
            df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)  # Skip problematic rows
            df.columns = df.columns.str.strip().str.lower()  # Normalize column names
            print(f"Columns in {file}: {df.columns.tolist()}")  # Debugging: Check column names
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        if 'event_id' in df.columns and 'event_type' in df.columns:
            df['event_id'] = pd.to_numeric(df['event_id'], errors='coerce')
            df = df.dropna(subset=['event_id'])
            df['event_id'] = df['event_id'].astype(int)
            
            # Store event rows
            event_rows.append(df)

print(f"Total CSVs processed: {len(event_rows)}")
if event_rows:
    event_data = pd.concat(event_rows, ignore_index=True)
    print(f"Total events collected: {len(event_data)}")

    event_counts = event_data['event_type'].value_counts()
    print("\nDistinct Event Types and Their Counts:")
    print(event_counts)
else:
    print("No valid event data found. Check column names in CSVs.")
