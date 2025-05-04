import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv('C:\Users\Varshith Daduvy\OneDrive\Desktop\IML-P\dataset.txt')
df.columns = df.columns.str.strip()  
print("Columns:", df.columns.tolist())
date_col = 'last_update'
pollutant_col = 'pollutant_avg'
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df.dropna(subset=[date_col, pollutant_col], inplace=True)
df['DayOfYear'] = df[date_col].dt.dayofyear
df['Month'] = df[date_col].dt.month
numeric_feats = ['DayOfYear', 'Month', 'latitude', 'longitude']
cat_feats = ['state', 'city', 'station', 'pollutant_id']
df.dropna(subset=numeric_feats + cat_feats, inplace=True)
numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_feats),
    ('cat', categorical_pipeline, cat_feats)
])
X = df[numeric_feats + cat_feats]
y = df[pollutant_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train_prep, y_train)
best_model = grid.best_estimator_
print("Best RF params:", grid.best_params_)
y_pred = best_model.predict(X_test_prep)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
joblib.dump(preprocessor, 'air_quality_preprocessor.pkl')
joblib.dump(best_model, 'air_quality_model.pkl')
def predict_air_quality():
    print("\nEnter details to predict air quality (pollutant average):")
    
    date_str = input("  Date (YYYY-MM-DD): ").strip()
    state = input("  State: ").strip()
    city = input("  City: ").strip()
    station = input("  Station: ").strip()
    pollutant = input("  Pollutant ID (e.g., PM2.5): ").strip()
    latitude = float(input("  Latitude: ").strip())
    longitude = float(input("  Longitude: ").strip())
    
    dt = pd.to_datetime(date_str)
    row = {
        'DayOfYear': dt.timetuple().tm_yday,
        'Month': dt.month,
        'state': state,
        'city': city,
        'station': station,
        'pollutant_id': pollutant,
        'latitude': latitude,
        'longitude': longitude
    }
    
    user_df = pd.DataFrame([row])
    X_u = preprocessor.transform(user_df)
    pred = best_model.predict(X_u)[0]
    print(f"\nPredicted Pollutant Average: {pred:.2f}")
predict_air_quality()
