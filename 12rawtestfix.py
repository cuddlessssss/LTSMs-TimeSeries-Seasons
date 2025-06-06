import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# 🕒 Start timer
total_start = time.time()

# 1. Load & Prepare Future Totals First
future_total_raw = pd.read_excel('future_total_sell_outs_pivoted.xlsx')
future_total_raw = future_total_raw[future_total_raw['Model Name'].str.strip() == 'K-55S30     MY1']
future_total_raw = future_total_raw[future_total_raw['Model Name'].str.contains("SENHENG ELECTRIC \(KL\) SDN BHD", case=False, na=False)]
future_total_melted = future_total_raw.melt(
    id_vars='Model Name', var_name='year_month', value_name='total_sell_out'
)
future_total_melted['year_month'] = pd.to_datetime(future_total_melted['year_month']).dt.to_period('M')

# 2. Load Past Sales Data
data = pd.read_excel('your_past_data.xlsx', sheet_name='updated')
data = data[data['Model Name'].str.strip() == 'K-55S30     MY1']
data.columns = data.columns.str.strip()
data = data[data['Category Description'].str.strip() == 'LCD TV']
data = data[data['Dealer Net Sell Out Qty'] >= 0]
data = data[data['Account Name'].str.contains("SENHENG ELECTRIC \(KL\) SDN BHD", case=False, na=False)]
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
            'forecast_qty': forecast_preds[month_idx, col_idx]
        })

raw_df = pd.DataFrame(raw_forecasts)
raw_df['Model Name'] = f1_encoder.inverse_transform(raw_df['model_enc'])
raw_df['Account Name'] = f2_encoder.inverse_transform(raw_df['account_enc'])

# Pivot to Excel-friendly format
raw_export_pivot = raw_df.pivot_table(index=['Model Name', 'Account Name'], columns='year_month', values='forecast_qty', fill_value=0)
raw_export_pivot.to_excel('raw_forecasts_no_scaling.xlsx')

# 🔢 Compare with Actuals (matching on month only)
actuals_data = pd.read_excel('your_past_data.xlsx', sheet_name='updated')
actuals_data.columns = actuals_data.columns.str.strip()
actuals_data = actuals_data[actuals_data['Category Description'].str.strip() == 'LCD TV']
actuals_data = actuals_data[actuals_data['Dealer Net Sell Out Qty'] >= 0]
actuals_data = actuals_data[actuals_data['Account Name'].str.contains("SENHENG ELECTRIC \(KL\) SDN BHD", case=False, na=False)]
actuals_data['Date'] = pd.to_datetime(actuals_data['Date'], errors='coerce')
actuals_data['month'] = actuals_data['Date'].dt.month

actual_agg = actuals_data.groupby(['Model Name', 'Account Name', 'month'])['Dealer Net Sell Out Qty'].sum().reset_index()
actual_agg.rename(columns={'Dealer Net Sell Out Qty': 'actual_qty'}, inplace=True)

forecast_vs_actual = raw_df.copy()
forecast_vs_actual['month'] = pd.to_datetime(forecast_vs_actual['year_month']).dt.month

comparison = pd.merge(forecast_vs_actual, actual_agg, on=['Model Name', 'Account Name', 'month'], how='left')
comparison['abs_error'] = (comparison['forecast_qty'] - comparison['actual_qty']).abs()
comparison['pct_error'] = comparison['abs_error'] / comparison['actual_qty'].replace(0, np.nan) * 100
comparison.sort_values(['Model Name', 'Account Name', 'year_month'], inplace=True)
comparison.to_excel('forecast_vs_actuals_comparison_month_only.xlsx', index=False)

# 📈 Plot trend line and actuals for one model-account if data overlaps
valid_data = comparison.dropna(subset=['actual_qty'])

if not valid_data.empty:
    sample_pair = valid_data.groupby(['Model Name', 'Account Name']).size().idxmax()
    sample_data = valid_data[(valid_data['Model Name'] == sample_pair[0]) & (valid_data['Account Name'] == sample_pair[1])]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_data['year_month'], sample_data['forecast_qty'], marker='o', label='Forecast')
    plt.plot(sample_data['year_month'], sample_data['actual_qty'], marker='x', label='Actual (Monthly Avg)')
    plt.title(f"Forecast vs Actual for {sample_pair[0]} - {sample_pair[1]} (Month-Matched)")
    plt.xlabel("Forecast Month")
    plt.ylabel("Sell-Out Qty")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast_vs_actual_monthonly_trend.png')
    plt.close()
else:
    print("⚠️ No overlapping forecast vs actual data found for plotting.")

explanation_notes = []
for _, row in comparison.iterrows():
    if not np.isnan(row['actual_qty']):
        if row['forecast_qty'] > row['actual_qty'] * 1.2:
            explanation_notes.append("Forecast was higher than usual — possibly due to an increasing trend detected in recent months or a strong seasonal effect.")
        elif row['forecast_qty'] < row['actual_qty'] * 0.8:
            explanation_notes.append("Forecast was lower — possibly due to recent downward trend or weaker seasonal pattern in training data.")
        else:
            explanation_notes.append("Forecast aligned reasonably well with historical average for this month.")
    else:
        explanation_notes.append("No actual data available for comparison.")

comparison['explanation'] = explanation_notes
comparison.to_excel('forecast_vs_actuals_comparison_month_only.xlsx', index=False)

print(f"✅ Finished in {time.time() - total_start:.2f} seconds")
