import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Fetch stock data (Here I used GRAB stock data)
df = yf.download('GRAB', start='2022-01-30', end='2025-01-30')

# 🔹 Fix MultiIndex issue
df.columns = ['_'.join(col).strip() for col in df.columns.values]

# Preprocessing
df.dropna(inplace=True)  # Drop any missing values
df = df[df['Volume_GRAB'] > 0]  # Remove rows where Volume = 0
df['Date'] = df.index
df.reset_index(drop=True, inplace=True)

# Feature Engineering - Moving Averages
df['SMA_50'] = df['Close_GRAB'].rolling(window=50).mean()
df['SMA_200'] = df['Close_GRAB'].rolling(window=200).mean()

# Ensure moving averages don't introduce NaNs before shifting
df.dropna(inplace=True)

# Debug: Check DataFrame shape before shifting
print("DataFrame shape before adding 'Target':", df.shape)

# Define Target variable (Next day's Close price)
df['Target'] = df['Close_GRAB'].shift(-1)

# Debug: Check if 'Target' is created
print("Columns in DataFrame before dropping NaN in 'Target':", df.columns)
print("Preview of 'Target':\n", df[['Close_GRAB', 'Target']].head(10))

# Drop rows where 'Target' is NaN
df.dropna(subset=['Target'], inplace=True)

# Features & Labels
X = df[['Open_GRAB', 'High_GRAB', 'Low_GRAB', 'Close_GRAB', 'Volume_GRAB', 'SMA_50', 'SMA_200']]
y = df['Target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')

# Predict Next Day's Closing Price
future_features = X.iloc[-1:].copy()
future_price = model.predict(future_features)

print(f"Predicted Next Day's Price: {future_price[0]:.2f}")
