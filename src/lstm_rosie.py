import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dill
import os

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

df_2019 = pd.read_csv('../run_ww_2019_w.csv')
df_2020 = pd.read_csv('../run_ww_2020_w.csv')


# convert 2019 objects to correct types
df_2019['datetime'] = pd.to_datetime(df_2019['datetime'], format='%Y-%m-%d')
df_2019['gender'] = df_2019['gender'].astype('category')
df_2019['age_group'] = df_2019['age_group'].astype('category')
df_2019['country'] = df_2019['country'].astype('category')
df_2019['major'] = df_2019['major'].astype('category')
df_2019.drop(columns=['Unnamed: 0'], inplace=True)

# convert 2020 objects to correct types
df_2020['datetime'] = pd.to_datetime(df_2020['datetime'], format='%Y-%m-%d')
df_2020['gender'] = df_2020['gender'].astype('category')
df_2020['age_group'] = df_2020['age_group'].astype('category')
df_2020['country'] = df_2020['country'].astype('category')
df_2020['major'] = df_2020['major'].astype('category')
df_2020.drop(columns=['Unnamed: 0'], inplace=True)

marathon_map = {
    'CHICAGO': '10-12',
    'BERLIN': '09-21',
    'LONDON': '04-27',
    'BOSTON': '04-21',
    'NEW YORK': '11-02'
}
from datetime import timedelta
df_2019['datetime'] = pd.to_datetime(df_2019['datetime'])

df_expanded = df_2019.copy()
df_expanded['major_split'] = df_expanded['major'].str.split(',')
df_expanded = df_expanded.explode('major_split')

df_expanded[['event', 'year']] = df_expanded['major_split'].str.extract(r'(\D+)\s+(\d{4})')
df_expanded['event'] = df_expanded['event'].str.strip()
df_expanded['year'] = df_expanded['year'].astype(int)
df_expanded['major_date'] = pd.to_datetime(
    df_expanded['year'].astype(str) + '-' + df_expanded['event'].map(marathon_map),
    errors='coerce'
)

one_month = pd.Timedelta(days=30)

# Check conditions
df_expanded['within-month-before'] = (
    (df_expanded['datetime'] > df_expanded['major_date'] - one_month) &
    (df_expanded['datetime'] <= df_expanded['major_date'])
)

df_expanded['within-month-after'] = (
    (df_expanded['datetime'] > df_expanded['major_date']) &
    (df_expanded['datetime'] <= df_expanded['major_date'] + one_month)
)

# Group back to original rows and aggregate using any()
df_result = df_expanded.groupby(df_expanded.index)[['within-month-before', 'within-month-after']].any()
df_result

df_2019 = df_2019.join(df_result)

df_2019_new = df_2019.pivot_table(
    index='athlete',
    columns='datetime',
    values=['distance', 'duration', 'within-month-before', 'within-month-after'],
    aggfunc='sum',
    fill_value=0
)
df_2019_new.columns = [
    f'{val}_week_{date.isocalendar()[1]}' for val, date in df_2019_new.columns
]


df_2019_new = df_2019_new.reset_index()
mask = ~df_2019['athlete'].duplicated()
df_2019_new['age_group'] = df_2019[mask]['age_group']
df_2019_new['country'] = df_2019[mask]['country']
df_2019_new['gender'] = df_2019[mask]['gender']
df_2019_new['major'] = df_2019[mask]['major']
age_map = {}
# compute mean age for each age group to convert to numeric
for age_group in df_2019_new['age_group'].unique():
    ages_split = age_group.split()
    mean_age = 0
    if ages_split[1] == '-':
        mean_age = (int(ages_split[0]) + int(ages_split[2])) / 2
    else:
        mean_age = (55 + 75) / 2
    age_map[age_group] = mean_age
df_2019_new['age_group'] = pd.Series(df_2019_new['age_group'].map(age_map), dtype=float)
df_2019_new = pd.get_dummies(df_2019_new, columns=['country'])
df_2019_new

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm

df = df_2019
df['week'] = df['datetime'].apply(lambda date: date.isocalendar()[1])
df['gender'] = df['gender'].eq('M')

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['athlete', 'datetime'])

# Drop any row with missing required values (or handle them differently)
df = df.dropna(subset=['distance', 'duration', 'gender', 'age_group', 'country'])


target_col = 'distance'
df['age_group'] = df['age_group'].apply(lambda x: age_map[x])

df = df.drop(columns=['country', 'major'])


lookback = 50
week = 51

# Filter your DataFrame
X = df[(df['week'] <= week) & (df['week'] >= week - lookback)]

# Get unique athletes
unique_athletes = X['athlete'].unique()
np.random.seed(42)
np.random.shuffle(unique_athletes)  # Shuffle to randomize the split

# Split indices
split_idx = int(len(unique_athletes) * 0.7)
train_athletes = unique_athletes[:split_idx]
test_athletes = unique_athletes[split_idx:]

# Create train and test sets based on athlete inclusion
X_train = X[X['athlete'].isin(train_athletes)]
X_test = X[X['athlete'].isin(test_athletes)]

feature_cols = [col for col in X_train.columns if col not in ['athlete', 'datetime', 'within-month-before', 'week',]]

X_seqs, y_targets = [], []
scalers = {}

for athlete_id, group in tqdm(X_train.groupby('athlete')):
    group = group.sort_values('datetime')
    if len(group) < lookback+1:
        print("BAD")
        continue
    
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(group[feature_cols])
    scalers[athlete_id] = scaler

    target_vals = group[target_col].values

    # Create non-overlapping sequences
    
    for i in range(0, len(group) - lookback + 2, lookback + 1):
        X_seqs.append(scaled[i:i+lookback])              # first 5
        y_targets.append(target_vals[i+lookback])        # 6th

# Convert to numpy arrays
X = np.array(X_seqs)  # shape: (samples, time_steps, features)
y = np.array(y_targets).reshape(-1, 1)

print(f"LSTM Input Shape: {X.shape}")  # (samples, time_steps, features)

# --- LSTM Model ---
model = Sequential()

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), activation='sigmoid'))
model.add(Dense(1))  # Output: predict next duration
model.compile(loss='mse', optimizer='adam')
model.summary()

# Train the model
model.fit(X, y, epochs=5, batch_size=64, verbose=1)

X_test_seqs = []
y_test_targets = []

for athlete_id, group in tqdm(X_test.groupby('athlete')):
    group = group.sort_values('datetime')
    if len(group) < lookback+1:
        print("BAD")
        continue
    
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(group[feature_cols])
    scalers[athlete_id] = scaler

    target_vals = group[target_col].values

    # Create non-overlapping sequences
    
    for i in range(0, len(group) - lookback + 2, lookback + 1):
        X_test_seqs.append(scaled[i:i+lookback])              # first 5
        y_test_targets.append(target_vals[i+lookback])        # 6th
        
X_test_input = np.array(X_test_seqs)
y_test_input = np.array(y_test_targets).reshape(-1, 1)
        
y_pred = model.predict(X_test_input)

from sklearn.metrics import mean_squared_error
import numpy as np

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_input, y_pred))
print(f"RMSE: {rmse}")

# Accuracy within multiple tolerance levels
tolerances = [round(t, 1) for t in np.arange(0.1, 1.1, 0.1)]

for tolerance in tolerances:
    within_tolerance = np.abs(y_test_input - y_pred) <= tolerance
    accuracy = np.mean(within_tolerance)
    print(f"Accuracy within ±{tolerance}: {accuracy * 100:.2f}%")
