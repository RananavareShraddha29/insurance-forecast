# Insurance Premium Price-Prediction


Introduction: This project analyzes and forecasts insurance premium values using historical data and time series modeling. The dataset initially contained 80 lakh (8 million) records, requiring preprocessing and aggregation to extract meaningful insights. The goal is to predict premium trends to aid insurers in pricing strategies and decision-making. Dataset link - https://www.kaggle.com/datasets/ex0ticone/brazilian-insurance-market-data/code

Data Processing

Grouping by Company Code, Product, and State to identify business trends. -Counting Premiums to analyze distribution.
Selecting Top Entities with the highest premium counts.
Reducing the dataset to 247 rows to focus on a particular company product for efficient modeling.
Time Series Analysis Stationarity tests, including the ADF test, revealed non-stationary data. To address this, differencing was applied to remove trends and improve model performance.

Predictive Modeling The following models were used:

ARIMA for non-seasonal trends.
SARIMA to incorporate seasonality.
Facebook Prophet, which excels in handling holidays, missing data, and seasonality.
Deployment The final step involves deploying the model for real-time predictions by:

Integrating with cloud services AWS for scalability.
Automating data updates for continuous improvement.
Conclusion This project provides a structured approach to insurance price forecasting, enhancing insurers' ability to optimize pricing strategies. Future steps involve refining models and deploying them for practical applications.
