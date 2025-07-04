# LTSMs-TimeSeries-Seasons
Makes Time-Based Predictions of future data by observing Trends and Seasonality
Based on inputs : Specified requirements of TOTALS of ALL Predictions for each month for EACH Model + Historical data 

Working ideas : 12backteststime2bba.py (NOT working on little data due to LTSMs learning seasonality STRICTLY) -> 12linear2.py (okay model) -> 12linear2another.py (almost complete as DECIDING between Lasso [many features prevent overfit] and Linear Regression -- amazing 
model fully captures Downtrend that 12linear2.py did not + after considering 12poscoefflinear.py problem of NOT considering downtrends) 

Final Product

-> 12linear2anotherlin.py (Linear Regression model using "Model Name" for total) -> 12linear2anotherlinmult.py (Same + Multi "Category Description" added) -> 12 linmultmodnum.py ("Model Code" instead of "Model Name" BUT ISSUE: "Model Code" starting with 0 -> 0s stripped in output!

or -> theone.py (12linear2another.py without the 99/1 percentiles, ALL data utilised)

or -> 12linearhuber.py (Huber regression replacing linear regression + remove 99/1 percentiles)

or RAW -> 12rawtestfix2.py / 12rawforecasts.py (RAW numbers forecast, BEFORE rounding, upscaling and colouring)

Test for Removing values beyond 1 and 99th Percentile -> SHOULD REMOVE!!
WHY? Cause we want forecasting power, NOT fit quality (overfitting)
The "Full" model seems to perfectly memorize the data (bad sign for forecasting).

ForwardTesting: Forecast 12 months ahead and find % of totals for BOTH Actuals + Forecast Months -> COMPARE SAME months
Output excel: forecast_output_trend_seasonality_testing_actuals.xlsx

---------------------------------------
IMPROVING IDEAS
The 99/1 percentile or Huber Regressions for different modelling
Filtering to only high data accounts








----------------------------------

[INFO] Average Errors Comparison:
Mean MAE (Trimmed): 0.2559
Mean MAE (Full):    ≈ 0
Mean MSE (Trimmed): 5.9981
Mean MSE (Full):    ≈ 0

| Trimmed Model                      | Untrimmed Model                     |
| ---------------------------------- | ----------------------------------- |
| Generalizes better                 | Overfits badly                      |
| Smoother forecasts                 | May react to past noise             |
| Slightly less perfect on past data | Nearly perfect fit (but misleading) |

🔍 Why the Regression Fit May Not Follow December Actuals:
Linear Regression Assumes Smooth Trends and Repeating Patterns
Your model uses a linear time trend and monthly seasonality via dummy variables.

It learns an “average” December effect based on the 12-month window.

If December 2024 had a sales spike (e.g., promotions, supply changes), it won’t be fully captured unless similar spikes occurred in previous Decembers.

Limited Historical Data (Only 12 Months)
With only one December in your dataset, the model has no other December to compare to.

This weakens the reliability of seasonal estimation for December.

Outlier Handling Trims Extremes
Your code removes values outside the 1st to 99th percentiles.

If December had an unusually high or low value, it might be partially excluded from the fit, reducing its influence.

No Special Holiday Effect Included
December might have retail effects (e.g., year-end sales, holidays).

Your model does not explicitly account for such event-based factors.


ALL Runtimes : Sub 1 Minute

# WHY did I not use LTSMs for this case? Pivoted to Trend + Seasonality with Linear Regression which captures LINEAR VS more DYNAMIC learning with Machine Learning!
Despite their greater capabilities to analyse seasonalities, I realise they were the main issue being hugely consuming on time + requiring more data to forecast 12 months.
This project is to build a model that can handle both long and SHORT term datasets too (below 12 months worth)

# WHY linear over lasso regression?
Lasso = Linear + penalising too heavy coeff -> aims to fix overfitting but it reduces flexibility of the model
Lasso is usually utilised when there are many features but our current dataset does NOT require it

1. Small Feature Set so we want to use all Coefficients in the Linear Regression Model

2. Captures Trend + Seasonality w/out Constraints

ALTERNATIVE Models BESIDES Linear Regression -> to handle #Non-Negative Outputs!

A. Log Transformation

Pros:

Prevents negative forecasts naturally.

Preserves downward trends (since log values still go down, just not below zero).

Stabilizes variance if your sales vary a lot.

Cons:

May underpredict when raw values are very low.

Concerns with accuracy trade-off as 1. Log compresses larger values and spreads out smaller ones.

Requires all input y >= 0.

B. sklearn.linear_model.PoissonRegressor

-> predicts only non-negative outputs

Pros:

Non-negativity is guaranteed mathematically.

More suitable for count data like sales.

Can still learn decreasing trends (e.g., 20 → 15 → 10).

Cons:

Assumes target is Poisson-distributed — may need to check if that's a good fit.

More sensitive to outliers than linear regression.

-----------------------------------------------

12linear2anotherlin.py (Linear Regression)

SAME as 12linear2another.py but changed regression model.

------------------------------------------------

12linear2another.py (Lasso Regression) = Linear Regression but for many features + solving overfitting issues. -> alpha set to 0.001

Purpose: Remove Restrictions on Downward Trend Learning in 12linear2.py where negative outputs -> 0 when its better to negative forecasts -> 0

SAME as 12linear2.py EXCEPT

Non-Negativity Logic has been shifted from within the model to outside of the model -> During Rescaling Step

Replacing Clipping Negative OUTPUTS

future_vals = np.clip(reg.predict(X_future), 0, None)

With Ignoring Negative FORECASTS to 0

preds_for_model = np.maximum(preds_for_model, 0)


------------------------------------------------
12linear2.py

Run Time : Less than a minute

What it does:

A linear regression model that maps out 1) Trends 2) Seasonality, with core focus on the trend.

Keeps only recent 24 months with filter, aggregating sales PER MONTH by (Model Name, Account Name), 

Linear Regression Modelling while ignoring extremes (beyond 1-99th percentile),

Rescaling is done at the Model Level, meaning ALL Account names for EACH Model Name

Firstly, we compute scaling factor through total model qty input/sum of predictions -> multiply all predictions by this 

Acquiring numbers with decimal points that add up total. 


Next, Use rounding logic, a) extract decimal points from integers b) round down all numbers get their total c) change some round down to round up starting with highest 

decimal point numbers until total in step b) matches input model total

Rounding to ensure 

1) Forecasted values are WHOLE NUMBERS, through clipping negative numbers to 0

2) Add up to totals given in 

future_total_sell_outs_pivoted.xlsx input requisites given, export into a fresh Excel File (forecast_output_trend_seasonality)

with rounded values in cells, coloured red = rounded UP, coloured green = rounded DOWN


Input Files: 

future_total_sell_outs_pivoted.xlsx (Columns B onwards:year_month ~ eg. 06-2025, Column A: Model Name, Values = monthly requirement EXACT totals to be met when splitting 

Model Name to all Account Name later!),

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

12poscoefflinear.py

LIMITATION: CANNOT learn downtrends, Biased towards Uptrends

Edit from 12linear.py

Different Non Negativity Logic

Differences

1) Using Lasso Machine Learning for L1 Regularisation of Coefficients, preventing overfitting

  Alpha value, default set to 0.001

  IF set to 0, it is a normal linear regression model!

3) No -ve Coeffs or Intercepts learnt in model. Minimum is 0.
----------------------------------------------

12backteststime2bba.py

LTSMs with sin/cos seasonality ideas

This way, your model can learn:

1. Seasonality (sin/cos of month)

2. Trend (year, time_index)

3. Month number for absolute time positioning

Key Problems:

1. 1st Month heavily skewed towards one Account for 2025-06,, Issue Attributed to 2024-06 input data learnt where ONLY that 1 account had a value

2. Point 1. Meaning yearly seasonality was forced... Which is why I pivoted to 
   
# Linear Regression in 12linear2.py

12backteststime2bb.py

Only using SEASONALITY to forecast forward 12 months of predictions
