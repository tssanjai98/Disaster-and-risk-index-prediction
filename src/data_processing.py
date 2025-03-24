import os
import pandas as pd

def load_and_merge_disaster_data(data_folder="../data/noaa_disaster_data", fema_data_path="../data/DisasterDeclarationsSummaries.csv"):
    us_state_to_abbrev = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
        'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
        'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD', 'MASSACHUSETTS': 'MA',
        'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO', 'MONTANA': 'MT',
        'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM',
        'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
        'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD',
        'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA',
        'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY'
    }

    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    df_list = []

    for file in sorted(csv_files):  
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path, low_memory=False)  
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df[["STATE", "STATE_FIPS", "YEAR", "EVENT_TYPE", "CZ_FIPS", "CZ_NAME", "BEGIN_DATE_TIME"]]

    merged_df["STATE"] = merged_df["STATE"].str.upper().map(us_state_to_abbrev)

    merged_df["STATE_FIPS"] = pd.to_numeric(merged_df["STATE_FIPS"], errors="coerce").astype("Int64")

    merged_df["BEGIN_DATE_TIME"] = pd.to_datetime(
        merged_df["BEGIN_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce"
    )

    merged_df["BEGIN_DATE_TIME"] = merged_df.apply(
        lambda row: row["BEGIN_DATE_TIME"].replace(year=row["YEAR"]) if pd.notnull(row["BEGIN_DATE_TIME"]) else None, axis=1
    )

    merged_df["BEGIN_DATE_TIME"] = merged_df["BEGIN_DATE_TIME"].dt.date

    merged_df = merged_df.dropna(subset=["STATE"])
    merged_df = merged_df[merged_df["CZ_FIPS"] != 0]
    merged_df = merged_df.drop_duplicates()


    fema_df = pd.read_csv(fema_data_path, low_memory=False)
    fema_df = fema_df[["state", "fipsStateCode", "fyDeclared", "incidentType", "fipsCountyCode", "designatedArea", "incidentBeginDate"]]

    fema_df["fipsStateCode"] = pd.to_numeric(fema_df["fipsStateCode"], errors="coerce").astype("Int64")
    fema_df = fema_df[fema_df["fipsCountyCode"] != 0]
    fema_df["incidentBeginDate"] = pd.to_datetime(fema_df["incidentBeginDate"]).dt.date

    event_mapping = {
        "Thunderstorm Wind": "Severe Storm",
        "Hail": "Severe Storm",
        "Winter Storm": "Winter Storm",
        "High Wind": "Severe Storm",
        "Drought": "Drought",
        "Flash Flood": "Flood",
        "Heavy Snow": "Snowstorm",
        "Tornado": "Tornado",
        "Flood": "Flood",
        "Heat": "Excessive Heat",
        "Strong Wind": "Severe Storm",
        "Excessive Heat": "Excessive Heat",
        "Blizzard": "Winter Storm",
        "Lightning": "Severe Storm",
        "Extreme Cold/Wind Chill": "Freezing",
        "Frost/Freeze": "Freezing",
        "Ice Storm": "Severe Ice Storm",
        "Wildfire": "Fire",
        "Tropical Storm": "Tropical Storm",
        "Coastal Flood": "Coastal Storm",
        "Debris Flow": "Mud/Landslide",
        "Hurricane (Typhoon)": "Hurricane",
        "Dust Storm": "Other",
        "Storm Surge/Tide": "Coastal Storm",
        "Avalanche": "Snowstorm",
        "Tsunami": "Tsunami",
        "Volcanic Ashfall": "Volcanic Eruption"
    }

    merged_df = merged_df[merged_df["EVENT_TYPE"].isin(event_mapping.keys())]
    merged_df["EVENT_TYPE"] = merged_df["EVENT_TYPE"].map(event_mapping)

    fema_df.rename(columns={
        "state": "STATE",
        "fipsStateCode": "STATE_FIPS",
        "fyDeclared": "YEAR",
        "incidentType": "EVENT_TYPE",
        "fipsCountyCode": "CZ_FIPS",
        "designatedArea": "CZ_NAME",
        "incidentBeginDate": "BEGIN_DATE_TIME"
    }, inplace=True)

    final_df = pd.concat([merged_df, fema_df], ignore_index=True)

    final_df = final_df.sort_values(by="CZ_NAME").drop_duplicates(
        subset=["STATE", "STATE_FIPS", "YEAR", "EVENT_TYPE", "CZ_FIPS", "BEGIN_DATE_TIME"], keep="first"
    )

    final_df.reset_index(drop=True, inplace=True)

    return final_df

