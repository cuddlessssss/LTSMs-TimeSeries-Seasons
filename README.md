# LTSMs-TimeSeries-Seasons
Makes Time-Based Predictions of future data by observing Trends and Seasonality
Based on inputs : Specified requirements of TOTALS of ALL Predictions for each month for EACH Model + Historical data 

------------------------------------------------
12linear2.py

Run Time : Less than a minute

What it does:

A linear regression model that maps out 1) Trends 2) Seasonality, with core focus on the trend.

Keeps only recent 24 months with filter, aggregating sales PER MONTH by (Model Name, Account Name), Modelling while ignoring extremes (PART 3 BELOW!), rescaling and rounding to ensure 1) Forecasted values are WHOLE NUMBERS 2) Add up to totals given in 

future_total_sell_outs_pivoted.xlsx input requisites given, export into a fresh Excel File (forecast_output_trend_seasonality)

with rounded values in cells, coloured red = rounded UP, coloured green = rounded DOWN


Input Files: future_total_sell_outs_pivoted.xlsx (Columns B onwards:year_month ~ eg. 06-2025, Column A: Model Name, Values = monthly requirement EXACT totals to be met when splitting Model Name to all Account Name later!),

your_past_data.xlsx (Historical data, columns: Date, Category Description (Filtered in code), Model Name, Account Name, Dealer Net Sell Out Qty



# 📌 Forecasting Logic Notes

"""
This code performs multivariate time series forecasting using a simple linear regression model per (Model Name, Account Name) pair.
It incorporates both trend and seasonality, with a post-processing step to ensure total volumes match provided forecasts.
"""

# ✅ 1. Data Preparation
- Filters sales data for "LCD TV" category and valid sell-out values.
- Converts dates to year-month periods.
- Keeps only the most recent 24 months.
- Encodes model and account names as integers.

# ✅ 2. Time Series Pivot
- Aggregates monthly sales by (Model, Account).
- Reshapes into a pivot table with time as rows and each (Model, Account) as a column.

# ✅ 3. Linear Regression Model with Seasonality
- For each (Model, Account) pair:
  - Uses month index (0–23) as the trend feature.
  - Adds monthly dummy variables (January–December) as seasonality indicators.
  - Removes outliers outside the 1st–99th percentile.
  - Trains a linear regression model on cleaned data.

# ✅ 4. Forecasting
- Predicts 12 future months by extending the trend and using correct month dummies.
- Clips negative values to zero.

# ✅ 5. Forecast Assembly
- Combines forecasts from all (Model, Account) pairs into a matrix.

# ✅ 6. Rescaling to Match Total Volumes
- Forecasts are rescaled to match known future model-level totals from external Excel.
- Rounds down forecasted values.
- Distributes any rounding remainder based on highest residuals.
- Marks adjustments as 'up' or 'down' for Excel highlighting.

# ✅ 7. Export to Excel
- Forecast is pivoted to (Model, Account) rows and months as columns.
- Excel cells are colored red if rounded up, green if rounded down.
- Final file: `forecast_output_trend_seasonality.xlsx`

# ✅ Summary
- Combines linear trend + monthly seasonality.
- Ignores short-term fluctuations (no windowed inputs).
- Removes extreme outliers.
- Forecasts are always adjusted to match external totals.

# 📌 End of Notes

----------------------------------------------

12backteststime2bba.py
This way, your model can learn:

1. Seasonality (sin/cos of month)

2. Trend (year, time_index)

3. Month number for absolute time positioning

12backteststime2bb.py

Only using SEASONALITY to forecast forward 12 months of predictions
