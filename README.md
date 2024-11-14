# Instacart Next Basket Prediction

## Overview

The goal of this project is to improve the basket prediction algorithm for Instacart, aiming to increase the F1 score from 0.25 to at least 0.28. Various techniques were used to explore customer purchasing behavior and enhance prediction accuracy. The final model surpassed the success threshold, achieving an F1 score of 0.30. Detailed insights and findings are available in the report.

## Objective

- Improve the F1 score of the current basket prediction algorithm by at least 0.03 (from 0.25 to 0.28).
- Analyze purchasing patterns and explore feature engineering to enhance decision-making.

## Dataset

The project uses a dataset containing approximately 30 million product orders from 3 million orders by 200,000 customers. The data includes multiple files detailing product information, user orders, and prior product orders.

## Frameworks and Tools

This project leverages a range of powerful frameworks and tools to ensure cutting-edge performance and efficiency. Here are the key technologies used:

### Core Technologies

- **Polars** ![Polars](https://img.shields.io/badge/Polars-9B5B3F?style=flat-square&logo=polars&logoColor=white)
- **PySpark** ![PySpark](https://img.shields.io/badge/PySpark-EE4C2C?style=flat-square&logo=apache-spark&logoColor=white)
- **XGBoost** ![XGBoost](https://img.shields.io/badge/XGBoost-FF9900?style=flat-square&logo=xgboost&logoColor=white)
- **LightGBM** ![LightGBM](https://img.shields.io/badge/LightGBM-F9A828?style=flat-square&logo=lightgbm&logoColor=black)
- **H2O** ![H2O](https://img.shields.io/badge/H2O-53B3F1?style=flat-square&logo=h2o&logoColor=white)

### Additional Tools
- **Plotly** ![Plotly](https://img.shields.io/badge/Plotly-3C0D3F?style=flat-square&logo=plotly&logoColor=white)
- **Pandas** ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
- **Matplotlib** ![Matplotlib](https://img.shields.io/badge/Matplotlib-003B57?style=flat-square&logo=matplotlib&logoColor=white)
- **Seaborn** ![Seaborn](https://img.shields.io/badge/Seaborn-9C27B0?style=flat-square&logo=seaborn&logoColor=white)

## Methodology

To tackle the problem, I used a combination of feature engineering, distributed computing, and GPU-accelerated training:

- **Feature Engineering**: Focused on user, product, and time-based features. Many engineered features performed exceptionally well.
- **Data Processing**: Initially used Polars for efficient data manipulation, then transitioned to PySpark for distributed data processing as the dataset grew in size.
- **Modeling**: Models were trained using **XGBoost**, **LightGBM**, and **H2O**, with distributed computing and GPU training for scalability.
- **Validation**: Employed a time-based validation strategy to ensure the model accounted for the sequential nature of purchases.

## Key Findings

1. **Reorder Patterns**: Users tend to reorder on the same day, the 7th day, or the 30th day after a previous order.![days_since_prior_order w.r.t count.png](https://github.com/d-sutariya/instacart_next_basket_prediction/blob/master/reports/figures/days_since_prior_order%20w.r.t%20count.png)
2. **Peak Ordering Time**: Orders are mostly placed between 8 AM and 4 PM.![order distribution w.r.t order hour of day](https://github.com/d-sutariya/instacart_next_basket_prediction/blob/master/reports/figures/order%20distribution%20w.r.t%20order%20hour%20of%20day.png)
4. **Product Preference**: Organic products are reordered 8% more frequently than non-organic products.![product type Organic vs Inorganic](https://github.com/d-sutariya/instacart_next_basket_prediction/blob/master/reports/figures/product%20type%20(Organic%20vs%20Inorganic).png)
5. **Department Reorder Rates**: Dairy, Eggs, Produce, Beverages, and Bakery have reorder rates above 65%, while Personal Care and Pantry have rates below 35%.![reorder percentage w.r.t department](https://github.com/d-sutariya/instacart_next_basket_prediction/blob/master/reports/figures/reorder%20percentage%20w.r.t%20department.png)

## Conclusion

The model achieved an F1 score of **0.30**, surpassing the success threshold of **0.27**. Future improvements can be made through further feature engineering and by exploring advanced architectures such as **LSTMs**, **GRUs**, and **Transformers** for better handling of sequential data.
