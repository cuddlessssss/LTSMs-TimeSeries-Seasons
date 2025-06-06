import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# 🕒 Start timer
total_start = time.time()

# 1. Load & Prepare Future Totals First
future_total_raw = pd.read_excel('future_total_sell_outs_pivoted.xlsx')
future_total_melted = future_total_raw.melt(
    id_vars='Model Name', var_name='year_month', value_name='total_sell_out'
)
future_total_melted['year_month'] = pd.to_datetime(future_total_melted['year_month']).dt.to_period('M')

# 2. Load Past Sales Data
data = pd.read_excel('your_past_data.xlsx', sheet_name='updated')
data.columns = data.columns.str.strip()
data = data[data['Category Description'].str.strip() == 'LCD TV']
data = data[data['Dealer Net Sell Out Qty'] >= 0]
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['year_month'] = data['Date'].dt.to_period('M')
data = data[data['Date'] >= data['Date'].max() - pd.DateOffset(months=10)]

# Label encode
f1_encoder = LabelEncoder()
f2_encoder = LabelEncoder()
data['Model Name_enc'] = f1_encoder.fit_transform(data['Model Name'])
data['Account Name_enc'] = f2_encoder.fit_transform(data['Account Name'])

# Use encoders on future totals
future_total_melted['model_enc'] = f1_encoder.transform(future_total_melted['Model Name'])

# 3. Preprocess Time Series Data
grouped = data.groupby(['year_month', 'Model Name_enc', 'Account Name_enc'])['Dealer Net Sell Out Qty'].sum().reset_index()
pivot = grouped.pivot_table(index='year_month', columns=['Model Name_enc', 'Account Name_enc'], values='Dealer Net Sell Out Qty', fill_value=0)
pivot = pivot.sort_index(axis=1)
column_index = pivot.columns.to_list()

# 4. Calculate trend + seasonality using linear regression with month dummies
trend_preds = {}
months_numeric = np.arange(len(pivot.index)).reshape(-1, 1)
month_dummies = pd.get_dummies(pivot.index.to_timestamp().month)
X_trend_season = np.hstack([months_numeric, month_dummies])

for col_idx, (model_enc, account_enc) in enumerate(pivot.columns):
    y = pivot.iloc[:, col_idx].values
    p1, p99 = np.percentile(y, [1, 99])
    mask = (y >= p1) & (y <= p99)

    reg = LinearRegression()
    reg.fit(X_trend_season[mask], y[mask])

    future_months = np.arange(len(pivot.index), len(pivot.index) + 12).reshape(-1, 1)
    last_month = pivot.index[-1].to_timestamp()
    future_periods = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), periods=12, freq='ME')
    future_month_nums = future_periods.month
    future_month_dummies = pd.get_dummies(future_month_nums)
    future_month_dummies = future_month_dummies.reindex(columns=month_dummies.columns, fill_value=0)
    X_future = np.hstack([future_months, future_month_dummies])

    future_vals = reg.predict(X_future)
    trend_preds[(model_enc, account_enc)] = future_vals

# 5. Assemble trend_preds into forecast_preds array
forecast_preds = np.zeros((12, len(pivot.columns)))
for idx, key in enumerate(pivot.columns):
    forecast_preds[:, idx] = trend_preds[key]

# ✨ Save raw forecast (unscaled, unrounded)
raw_forecasts = []
for month_idx in range(forecast_preds.shape[0]):
    forecast_month = pivot.index[-1] + (month_idx + 1)
    for col_idx, (model_enc, account_enc) in enumerate(column_index):
        raw_forecasts.append({
            'year_month': forecast_month.to_timestamp(),
            'model_enc': model_enc,
            'account_enc': account_enc,
            'raw_forecast_qty': forecast_preds[month_idx, col_idx]
        })

raw_df = pd.DataFrame(raw_forecasts)
raw_df['Model Name'] = f1_encoder.inverse_transform(raw_df['model_enc'])
raw_df['Account Name'] = f2_encoder.inverse_transform(raw_df['account_enc'])

# Pivot to Excel-friendly format
raw_export_pivot = raw_df.pivot_table(index=['Model Name', 'Account Name'], columns='year_month', values='raw_forecast_qty', fill_value=0)

# Save to Excel
raw_export_pivot.to_excel('raw_forecasts_no_scaling.xlsx')

print(f"✅ Finished in {time.time() - total_start:.2f} seconds")
