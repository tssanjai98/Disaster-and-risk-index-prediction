import pandas as pd
import os

EVENT_TYPE_MAPPING = {
    "ice storm": "severe ice storm",
    "flood": "flood",
    "tornado": "tornado",
    "drought": "drought",
    "thunderstorm wind": "severe storm",
    "winter storm": "winter storm",
    "debris flow": "mud/landslide",
    "wildfire": "fire",
    "hurricane (typhoon)": "hurricane",
    "tropical storm": "tropical storm",
    "strong wind": "severe storm",
    "storm surge/tide": "coastal storm",
    "frost/freeze": "freezing",
    "volcanic ash": "volcanic eruption",
    "tsunami": "tsunami"
}

def load_noaa_datasets(data_dir="../data/noaa_disaster_data"):
    """
    Reads all NOAA disaster datasets from the given directory and returns a combined dataframe.
    Filters datasets that contain 'event_id' and 'event_type' columns.
    Ensures 'event_id' is numeric and removes invalid entries.
    """
    event_rows = []

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
                df.columns = df.columns.str.strip().str.lower()  # Normalize column names
            except Exception as e:
                continue

            if 'event_id' in df.columns and 'event_type' in df.columns:
                df['event_id'] = pd.to_numeric(df['event_id'], errors='coerce')
                df = df.dropna(subset=['event_id'])
                df['event_id'] = df['event_id'].astype(int)

                event_rows.append(df)
    
    if event_rows:
        event_data = pd.concat(event_rows, ignore_index=True)
        return event_data
    else:
        return pd.DataFrame()

def convert_time_column(time_col):
    """Convert NOAA time from integer format (e.g., 1922) to HH:MM format."""
    time_col = pd.to_numeric(time_col, errors="coerce")
    return time_col.apply(lambda x: f"{int(x // 100):02}:{int(x % 100):02}" if pd.notna(x) else None)

def fill_missing_end_date(row, median_durations):
    """Fill missing end dates based on median duration per state and event type."""
    if pd.isna(row["END_DATE"]):
        median_duration = median_durations.get((row["STATE_ABBR"], row["EVENT_TYPE"]), 1)
        median_duration = 1 if pd.isna(median_duration) else int(median_duration)
        return row["BEGIN_DATE"] + pd.Timedelta(days=median_duration)
    return row["END_DATE"]

def load_combine_datasets():
    noaa_dir = "../data/noaa_disaster_data"
    fema_file = "../data/DisasterDeclarationsSummaries.csv"

    # Load FEMA dataset
    fema_df = pd.read_csv(fema_file, low_memory=False)
    fema_df = fema_df[["state", "incidentBeginDate", "incidentEndDate", "incidentType"]]
    fema_df.rename(columns={
        "state": "STATE_ABBR",
        "incidentBeginDate": "BEGIN_DATE",
        "incidentEndDate": "END_DATE",
        "incidentType": "EVENT_TYPE"
    }, inplace=True)

    fema_df["BEGIN_DATE"] = pd.to_datetime(fema_df["BEGIN_DATE"], errors="coerce").dt.tz_localize(None)
    fema_df["END_DATE"] = pd.to_datetime(fema_df["END_DATE"], errors="coerce").dt.tz_localize(None)

    fema_df["DURATION"] = (fema_df["END_DATE"] - fema_df["BEGIN_DATE"]).dt.days
    median_durations = fema_df.groupby(["STATE_ABBR", "EVENT_TYPE"])["DURATION"].median()

    fema_df["END_DATE"] = fema_df.apply(lambda row: fill_missing_end_date(row, median_durations), axis=1)
    fema_df.drop(columns=["DURATION"], inplace=True)

    fema_df["BEGIN_TIME"] = None
    fema_df["END_TIME"] = None

    fema_df["STATE_ABBR"] = fema_df["STATE_ABBR"].str.strip().str.upper()
    fema_df["EVENT_TYPE"] = fema_df["EVENT_TYPE"].str.strip().str.lower()

    noaa_df = load_noaa_datasets(noaa_dir)

    if not noaa_df.empty:
        relevant_columns = ["state_abbr", "begin_date", "begin_time", "end_date", "end_time", "event_type"]
        available_columns = [col for col in relevant_columns if col in noaa_df.columns]
        noaa_df = noaa_df[available_columns]

        column_mapping = {
            "state_abbr": "STATE_ABBR",
            "begin_date": "BEGIN_DATE",
            "begin_time": "BEGIN_TIME",
            "end_date": "END_DATE",
            "end_time": "END_TIME",
            "event_type": "EVENT_TYPE"
        }
        noaa_df.rename(columns=column_mapping, inplace=True)

        noaa_df["BEGIN_DATE"] = pd.to_datetime(noaa_df["BEGIN_DATE"], errors="coerce").dt.tz_localize(None)
        noaa_df["END_DATE"] = pd.to_datetime(noaa_df["END_DATE"], errors="coerce").dt.tz_localize(None)

        noaa_df["STATE_ABBR"] = noaa_df["STATE_ABBR"].str.strip().str.upper()
        noaa_df["EVENT_TYPE"] = noaa_df["EVENT_TYPE"].str.strip().str.lower()

        noaa_df["BEGIN_TIME"] = convert_time_column(noaa_df["BEGIN_TIME"]) if "BEGIN_TIME" in noaa_df else None
        noaa_df["END_TIME"] = convert_time_column(noaa_df["END_TIME"]) if "END_TIME" in noaa_df else None

        noaa_df["EVENT_TYPE"] = noaa_df["EVENT_TYPE"].map(EVENT_TYPE_MAPPING).fillna(noaa_df["EVENT_TYPE"])

    combined_df = pd.concat([noaa_df, fema_df], ignore_index=True)

    combined_df.drop_duplicates(inplace=True)
    combined_df = combined_df[combined_df["STATE_ABBR"] != "XX"]

    return combined_df