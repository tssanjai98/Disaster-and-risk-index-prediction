import os
import re
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBRegressor
import joblib
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from geopy.geocoders import Nominatim
import json
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from data_processing import load_and_merge_disaster_data

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
RESULT_PATH = "../results/forecast_2025_results.csv"
CLEANED_RESULT_PATH = "../results/all_incidents_2025.csv"

LSTM_MODEL_DIR = "../models/lstm"
XGB_MODEL_DIR = "../models/xgboost"
PROPHET_MODEL_DIR = "../models/prophet"

os.makedirs(LSTM_MODEL_DIR, exist_ok=True)
os.makedirs(XGB_MODEL_DIR, exist_ok=True)
os.makedirs(PROPHET_MODEL_DIR, exist_ok=True)

def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:"*?<>|]+', '', name)
    name = name.replace(' ', '_')
    return name

def create_sequences(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def get_coordinates(state_name, geolocator, coord_cache):
    if state_name in coord_cache:
        return coord_cache[state_name]
    try:
        loc = geolocator.geocode(state_name)
        if loc:
            coord_cache[state_name] = (loc.latitude, loc.longitude)
            return loc.latitude, loc.longitude
    except:
        pass
    coord_cache[state_name] = (np.nan, np.nan)
    return np.nan, np.nan

def process_group(event_type, state, group, state_encoder_classes):
    time_step = 5
    xgb_features = ['month', 'year', 'lstm_predictions', 'state_encoded']

    geolocator = Nominatim(user_agent="disaster_forecaster")
    coord_cache = {}

    monthly = group.groupby(group['BEGIN_DATE_TIME'].dt.to_period('M')).size().reset_index(name='count')
    monthly['ds'] = monthly['BEGIN_DATE_TIME'].dt.to_timestamp()
    series = monthly[['ds', 'count']].rename(columns={'count': 'y'})

    if len(series) < max(time_step + 2, 12):
        return None

    safe_event = safe_filename(event_type)
    safe_state = safe_filename(state)

    model_path = os.path.join(LSTM_MODEL_DIR, f"{safe_event}_{safe_state}_model.keras")
    data = series['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    train_size = int(len(scaled) * 0.8)
    train, test = scaled[:train_size], scaled[train_size:]
    X_train, y_train = create_sequences(train, time_step)
    X_test, y_test = create_sequences(test, time_step)

    if X_test.size == 0:
        return None

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    if os.path.exists(model_path):
        lstm_model = load_model(model_path)
    else:
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        lstm_model.save(model_path)

    lstm_pred = lstm_model.predict(X_test)
    lstm_pred_rescaled = scaler.inverse_transform(lstm_pred)

    model_path = os.path.join(XGB_MODEL_DIR, f"{safe_event}_{safe_state}_model.pkl")
    df_xgb = series.copy()
    df_xgb['month'] = df_xgb['ds'].dt.month
    df_xgb['year'] = df_xgb['ds'].dt.year

    encoder = LabelEncoder()
    encoder.classes_ = state_encoder_classes
    df_xgb['state_encoded'] = encoder.transform([state] * len(df_xgb))
    df_xgb['lstm_predictions'] = np.nan
    df_xgb.loc[len(df_xgb) - len(lstm_pred):, 'lstm_predictions'] = lstm_pred_rescaled.flatten()

    train_xgb = df_xgb[:train_size]
    test_xgb = df_xgb[train_size:]

    if os.path.exists(model_path):
        xgb_model = joblib.load(model_path)
    else:
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(train_xgb[xgb_features], train_xgb['y'])
        joblib.dump(xgb_model, model_path)

    xgb_pred = xgb_model.predict(test_xgb[xgb_features])

    model_path = os.path.join(PROPHET_MODEL_DIR, f"{safe_event}_{safe_state}_model.json")
    if os.path.exists(model_path):
        with open(model_path, 'r') as fin:
            model_json = json.load(fin)
            if isinstance(model_json, dict):
                os.remove(model_path)

    if os.path.exists(model_path):
        with open(model_path, 'r') as fin:
            prophet_model = model_from_json(json.load(fin))
    else:
        prophet_model = Prophet()
        prophet_model.fit(series)
        with open(model_path, 'w') as fout:
            json.dump(model_to_json(prophet_model), fout)

    future = prophet_model.make_future_dataframe(periods=12, freq='MS')
    forecast = prophet_model.predict(future)
    prophet_2025 = forecast[forecast['ds'].dt.year == 2025][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    prophet_2025.columns = ['Date', 'Prophet Forecast', 'Prophet Lower', 'Prophet Upper']

    future_dates = pd.date_range(start='2025-01-01', periods=12, freq='MS')
    seq = scaled[-time_step:]
    future_lstm = []

    for _ in range(12):
        seq = seq.reshape(1, time_step, 1)
        next_pred = lstm_model.predict(seq)
        future_lstm.append(next_pred[0, 0])
        seq = np.append(seq[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

    future_lstm = scaler.inverse_transform(np.array(future_lstm).reshape(-1, 1)).flatten()

    future_df = pd.DataFrame({
        'Date': future_dates,
        'month': future_dates.month,
        'year': future_dates.year,
        'lstm_predictions': future_lstm,
        'state_encoded': encoder.transform([state] * 12)
    })

    xgb_2025 = xgb_model.predict(future_df[xgb_features])
    hybrid = (future_lstm + xgb_2025) / 2

    combined = pd.DataFrame({
        'Date': future_dates,
        'Incident Type': event_type,
        'State': state,
        'LSTM Forecast': future_lstm,
        'XGBoost Forecast': xgb_2025,
        'Hybrid Forecast': hybrid
    })

    final = combined.merge(prophet_2025, on='Date')
    lat, lon = get_coordinates(state, geolocator, coord_cache)
    final['Latitude'] = lat
    final['Longitude'] = lon

    return final

def main():
    if os.path.exists(RESULT_PATH):
        os.remove(RESULT_PATH)

    df = load_and_merge_disaster_data()
    df['BEGIN_DATE_TIME'] = pd.to_datetime(df['BEGIN_DATE_TIME'])
    df['year'] = df['BEGIN_DATE_TIME'].dt.year
    df['month'] = df['BEGIN_DATE_TIME'].dt.month
    df['day'] = df['BEGIN_DATE_TIME'].dt.day

    encoder = LabelEncoder()
    df['state_encoded'] = encoder.fit_transform(df['CZ_NAME'])
    state_encoder_classes = encoder.classes_

    grouped = df.groupby(['EVENT_TYPE', 'CZ_NAME'])

    args_list = [
        (event_type, state, group.copy(), state_encoder_classes)
        for (event_type, state), group in grouped
    ]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.starmap(process_group, args_list), total=len(args_list), desc="Forecasting"))

    all_results = [df for df in results if df is not None]
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(RESULT_PATH, index=False)

        final_df['Date'] = pd.to_datetime(final_df['Date'], errors='coerce')
        final_df = final_df.groupby(['Incident Type', 'State', 'Date']).first().reset_index()
        final_df.to_csv(CLEANED_RESULT_PATH, index=False)
        print(f"Saved cleaned forecast results to: {CLEANED_RESULT_PATH}")
    else:
        print("No valid forecasts generated.")

if __name__ == "__main__":
    main()