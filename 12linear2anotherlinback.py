import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load your past sales data
data = pd.read_excel('your_past_data.xlsx', sheet_name='updated')
data.columns = data.columns.str.strip()

# Filter: only LCD TV and non-negative sales
data = data[data['Category Description'].str.strip() == 'LCD TV']
data = data[data['Dealer Net Sell Out Qty'] >= 0]

# Add year_month column and filter to last 24 months
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['year_month'] = data['Date'].dt.to_period('M')
data = data[data['Date'] >= data['Date'].max() - pd.DateOffset(months=24)]

# Group and pivot
grouped = data.groupby(['year_month', 'Model Name', 'Account Name'])['Dealer Net Sell Out Qty'].sum().reset_index()
pivot = grouped.pivot_table(index='year_month', columns=['Model Name', 'Account Name'], values='Dealer Net Sell Out Qty', fill_value=0)
pivot = pivot.sort_index()

# Filter out model-account combinations with total sales < threshold (e.g. 30 units over 24 months)
volume_filter = pivot.sum(axis=0) >= 30
pivot = pivot.loc[:, volume_filter]

# Train-test split (last 3 months as test)
train_pivot = pivot.iloc[:-3]
test_pivot = pivot.iloc[-3:]

# Time and month dummy features
month_dummies_train = pd.get_dummies(train_pivot.index.to_timestamp().month)
X_train = np.hstack([np.arange(len(train_pivot)).reshape(-1, 1), month_dummies_train])

month_dummies_test = pd.get_dummies(test_pivot.index.to_timestamp().month)
month_dummies_test = month_dummies_test.reindex(columns=month_dummies_train.columns, fill_value=0)
X_test = np.hstack([np.arange(len(train_pivot), len(train_pivot) + len(test_pivot)).reshape(-1, 1), month_dummies_test])

# Backtest per column
results = []

for col in train_pivot.columns:
    y_train = train_pivot[col].values
    y_test = test_pivot[col].values

    if np.all(y_test == 0):
        continue

    # Trim training outliers
    p1, p99 = np.percentile(y_train, [1, 99])
    mask = (y_train >= p1) & (y_train <= p99)

    model = LinearRegression()
    model.fit(X_train[mask], y_train[mask])
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results.append({
        'Model Name': col[0],
        'Account Name': col[1],
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    })

# Output results
results_df = pd.DataFrame(results)
print("\n🔍 Backtest Results (Filtered for ≥30 total units over 24 months):")
print(results_df.groupby('Model Name')[['MAE', 'MSE', 'RMSE']].mean().round(2))

# Export full table
results_df.to_excel("backtest_results_rmse_filtered.xlsx", index=False)
