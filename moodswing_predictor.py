import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

data_path = 'Dataset/mcphases_data/'

#Getting self-reported symptoms and cycle info
df_selfreport = pd.read_csv(data_path + 'hormones_and_selfreport.csv')
df_selfreport = df_selfreport.rename(columns={'id': 'subject_id', 'day_in_study': 'Time_Index'})

core_cols = [
    'subject_id', 
    'Time_Index', 
    'phase',                # Cycle Phase (for one-hot encoding)           
    'stress',               # Symptom
    'cramps',               # Symptom
    'headaches',            # Symptom
    'fatigue',              # Symptom
    'exerciselevel',        # Symptom
    'moodswing'             # Target
]

#Filtering the main dataframe to keep only the features listed above
df_core = df_selfreport[core_cols].copy()

cols_to_convert = ['stress', 'cramps', 'headaches', 'fatigue', 'moodswing']

#Translating severity strings into a 0-5 numerical scale.
severity_mapping = {
    'Not at all': 0,
    'Very Low/Little': 1,
    'Low': 2,
    'Moderate': 3,
    'High': 4,
    'Very High': 5
}

#Exercise scale is slightly different so it needs its own severity map
severity_mapping_exercise = {
    'Very Low': 1,
    'Low': 2,
    'Moderate': 3,
    'High': 4,
    'Very High': 5
}

#Applying the severity mapping
for col in cols_to_convert:
    df_core[col] = df_core[col].map(severity_mapping)
    df_core[col] = pd.to_numeric(df_core[col], errors='coerce')

df_core['exerciselevel'] = df_core['exerciselevel'].map(severity_mapping_exercise)
df_core['exerciselevel'] = pd.to_numeric(df_core['exerciselevel'], errors='coerce')

#Loading in sleep data
df_sleep = pd.read_csv(data_path + 'sleep.csv')
df_sleep = df_sleep.rename(columns={'id': 'subject_id', 'sleep_end_day_in_study': 'Time_Index'})
sleep_cols = ['subject_id', 'Time_Index', 'minutesasleep']
df_sleep_daily = df_sleep[sleep_cols].copy()


#Handling potential multiple sleep records by averaging the total_sleep_minutes for that day
df_sleep_daily = df_sleep_daily.groupby(['subject_id', 'Time_Index']).mean().reset_index()


#Merging the 2 dataframes
df_merged = pd.merge(
    df_core,
    df_sleep_daily,
    on=['subject_id', 'Time_Index'],
    how='left' 
)

df_merged = df_merged.set_index(['subject_id', 'Time_Index'])


#Converting the 'phase' (like Follicular, Luteal) text into numerical features for the model using one-hot encoding
df_features = df_merged.copy()
df_features = pd.get_dummies(df_features, columns=['phase'], drop_first=True, prefix='Phase')

print("Missing values before imputation:")
print(df_features.isna().sum())

df_features_sorted = df_features.sort_index()

#linear interpolation grouped by subject to fill in missing data for each person
df_clean_imputed = df_features_sorted.groupby(level='subject_id').apply(
    lambda x: x.interpolate(method='linear')
)

print("Missing values after imputation:")
print(df_clean_imputed.isna().sum())

#drop any remaining rows with NaN (rows where interpolation failed)
df_model_data = df_clean_imputed.copy().dropna()

#we want the model to see what happened on previous days. These are the features we want to "look back" at.
features_to_lag = [
    'stress',                 
    'cramps', 
    'headaches', 
    'fatigue', 
    'exerciselevel',
    'minutesasleep', 
    'moodswing'
]

lag_periods = [1,3, 7]
for feature in features_to_lag:
    for lag in lag_periods:
        #group by subject before shifting to ensure lag is subject-specific
        df_model_data[f'{feature}_Lag_{lag}D'] = df_model_data.groupby(level=0)[feature].shift(lag)


#final cleanup: drop the initial rows that are NaN due to the lagging
df_model_ready = df_model_data.dropna()


#ridge regression

#defining our target Y and features X.
target = 'moodswing' 
Y = df_model_ready[target]
X = df_model_ready.drop(target, axis=1).reset_index(drop=True)

#Splitting data into a training set and a test set (20% of data)
#since this is a time series, we do not shuffle the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False 
)

model = Ridge(alpha=1.0) 
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("\n____Model Performance____")
print(f"Mean Squared Error on Test Set: {mse:.4f}")
print(f"Root Mean Squared Error on Test Set: {rmse:.4f}")

print(f"\n____Predictive Relationships (Top 10 Coefficients)____")
coefficients_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print(coefficients_df.round(4).head(10))


#Saving the trained model for future use and expansion of project
model_filename = 'moodswing_predictor_ridge_model.joblib'
joblib.dump(model, model_filename)