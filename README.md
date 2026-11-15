# Indian Electricity Day-Ahead Market Price Forecasting

**A machine learning approach to forecasting 15-minute electricity market clearing prices in India's Day-Ahead Market (DAM).**

## Overview

Electricity prices fluctuate throughout the day due to changes in demand, supply, market participation, and other factors. Accurate day-ahead price forecasting can help market participants understand price patterns and make more informed decisions.

This project develops a machine learning pipeline to forecast **15-minute Market Clearing Prices (MCP)** in the Indian Electricity Exchange (IEX) Day-Ahead Market.

Using historical 2025 IEX market data, the project explores price patterns, engineers time-series features, and compares machine learning models against simple historical forecasting baselines.


## Objectives

- Analyze historical IEX Day-Ahead Market price patterns.
- Identify daily, weekly, and monthly electricity price trends.
- Develop time-based and historical price features.
- Build machine learning models for 15-minute day-ahead price forecasting.
- Compare ML models against historical forecasting baselines.
- Evaluate model performance using chronological data splits.

## Dataset

**Source:** [Indian Energy Exchange (IEX) – Day-Ahead Market](https://www.iexindia.com/market-data/day-ahead-market/market-snapshot)

The dataset contains historical Day-Ahead Market observations covering January–December 2025.

| Attribute | Description |
|---|---|
| Market | Indian Day-Ahead Market (DAM) |
| Time resolution | 15 minutes |
| Coverage | January–December 2025 |
| Total observations | 35,040 |
| Target variable | Market Clearing Price (MCP) |
| Target unit | INR/MWh |

### Dataset Features

| Feature | Description |
|---|---|
| `date` | Delivery date |
| `period` | 15-minute delivery interval |
| `purchase_bid` | Purchase bid volume |
| `sell_bid` | Sell bid volume |
| `mcv` | Market Clearing Volume |
| `final_scheduled_volume` | Final scheduled electricity volume |
| `mcp` | Market Clearing Price |
| `congestion` | Market congestion indicator |

The raw dataset was manually collected from the IEX Market Snapshot dashboard and processed using Python.

*Note: The dataset is not included in this repository. Refer to the IEX source for data access and availability. Redistribution and usage are subject to the source's applicable terms.*

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the distribution and temporal behavior of electricity prices.

Key observations:

- **Evening price peaks:** Prices generally increase during evening hours, with the highest average prices observed around 19:00.
- **Afternoon price dips:** Lower average prices were observed around 12:00–13:00.
- **Weekly patterns:** Sundays generally showed lower prices than weekdays.
- **Price variability:** The dataset contains both normal and extreme price observations, making accurate forecasting challenging.

EDA included:

- Price distribution and boxplots.
- Full-year price trends.
- Daily and monthly average prices.
- Hourly price patterns.
- Weekday versus weekend analysis.
- Hour-of-day and weekday heatmaps.
- Correlation analysis.
- High-price observation analysis.

## Methodology

The project follows a chronological machine learning workflow.

### 1. Data Preprocessing

- Standardized column names.
- Validated 15-minute delivery intervals.
- Converted numerical columns into appropriate data types.
- Parsed dates and delivery periods into timestamps.
- Removed duplicate timestamps and missing target observations.
- Converted congestion indicators into numerical values.
- Created date and time attributes.

Data validation confirmed 35,040 observations, with no duplicate timestamps or missing 15-minute intervals.

### 2. Feature Engineering

The following features were developed:

**Temporal features**
- Hour and 15-minute time slot.
- Day of the week.
- Month.
- Weekend indicator.

**Cyclical encoding**

Cyclical encoding was applied to represent repeating time patterns:

- `time_sin`, `time_cos`
- `dow_sin`, `dow_cos`
- `month_sin`, `month_cos`

**Historical price features**
- `lag_1_day`: Price from the corresponding 15-minute interval on the previous day.
- `lag_7_days`: Price from the corresponding interval seven days earlier.
- `rolling_7d_mean`: Mean of daily average prices over the previous seven complete days.
- `rolling_7d_std`: Standard deviation of daily average prices over the previous seven complete days.

Historical features were constructed to use information available before the delivery day, supporting the day-ahead forecasting objective.

### 3. Forecasting Strategy

The forecasting objective is to predict all **96 fifteen-minute price intervals for the next delivery day**, using information available before that delivery day begins.

The prediction features therefore use historical market observations and calendar information rather than actual prices from the target delivery day.

### 4. Chronological Data Splitting

The dataset was divided chronologically to avoid random mixing of past and future observations.

| Dataset | Period | Purpose |
|---|---|---|
| Training | January–September 2025 | Model training |
| Validation | October–November 2025 | Model comparison and hyperparameter experimentation |
| Evaluation | December 2025 | Performance evaluation |

The selected models were retrained using January–November data before the December evaluation.

**Evaluation note:** Hyperparameter experimentation was also performed using December results. Consequently, December should be interpreted as a post-tuning evaluation rather than a completely untouched final test set.

## Models

Four forecasting approaches were compared.

### Baseline Models

1. **Previous Day:** Uses the previous day's price for the corresponding 15-minute interval.
2. **Previous Week:** Uses the price from the corresponding interval seven days earlier.

### Machine Learning Models

3. **Random Forest Regressor**
4. **XGBoost Regressor**

Both machine learning models were trained using engineered temporal and historical price features.

## Model Performance

The following results represent the December 2025 evaluation after hyperparameter experimentation.

### Overall Performance

| Model | MAE (₹/MWh) | RMSE (₹/MWh) | R² |
|---|---:|---:|---:|
| **XGBoost** | **349.73** | **623.45** | **0.92** |
| Random Forest | 351.09 | 650.77 | 0.91 |
| Previous Day | 532.58 | 1,180.17 | 0.71 |
| Previous Week | 559.54 | 1,054.96 | 0.77 |

XGBoost achieved the best overall performance.

Compared with the Previous Day baseline:

- MAE decreased by approximately **34.3%**.
- RMSE decreased by approximately **47.2%**.
- R² increased from 0.71 to 0.92.

### Performance by Price Category

Actual prices were grouped into three categories for error analysis.

| Category | Price Range (₹/MWh) | Observations |
|---|---|---:|
| Normal | Below 4,000 | 2,105 |
| High | 4,000–8,000 | 581 |
| Very High | 8,000 and above | 290 |

**Category-wise MAE**

| Model | Normal | High | Very High |
|---|---:|---:|---:|
| Previous Day | 317.35 | 961.80 | 1,234.99 |
| Previous Week | 347.17 | 1,038.85 | 1,140.75 |
| Random Forest | 200.56 | **652.07** | 840.78 |
| XGBoost | **196.32** | 665.90 | **829.81** |

XGBoost performs best in the normal and very-high-price categories, while Random Forest performs best in the high-price category.

Extreme price observations remain more challenging to forecast than normal market conditions.




## Limitations and Future Improvements

Although the models demonstrate promising results, several improvements could be explored:

- Incorporating electricity demand and supply-side information.
- Integrating weather and renewable energy generation data.
- Evaluating performance on additional years of historical data.
- Using walk-forward validation for more robust time-series evaluation.
- Investigating methods for improving extreme price forecasting.
- Exploring LightGBM, CatBoost, and deep learning approaches.
- Developing a forecasting API or interactive dashboard for daily predictions.

## Conclusion

This project demonstrates how machine learning can be applied to short-term electricity price forecasting using historical Indian electricity market data.

The experiments show that Random Forest and XGBoost can outperform simple historical-price forecasting baselines. XGBoost achieved the strongest overall December evaluation results, with an MAE of ₹349.73/MWh and an R² of 0.92.

The project also highlights the challenges of forecasting high and extreme electricity prices and the importance of chronological validation and leakage-aware feature engineering.
