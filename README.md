# Used Car Market Analysis
<br></br>

## 1. Problem Statement
Used car prices vary widely across dealerships, brands, and vehicle conditions. This project explores how structured vehicle attributes such as age, mileage, and engine size, shape market value. The goal is to build predictive models that estimate fair prices and classify vehicles into meaningful age groups.


## 2. Why This Matters
Accurate price estimation supports better decision‑making for buyers, sellers, and dealerships. Understanding which features drive value helps reduce pricing uncertainty, improve inventory strategy, and increase transparency in a market where information is often unevenly distributed.

## 3. Data Source
Both projects use publicly available datasets from Kaggle, each containing dealership listings with structured vehicle attributes. Although the datasets differ in size and scope, they share common features such as year, mileage, engine size, and price—allowing for consistent modeling across contexts.

## 4. Modeling Approach
* Data cleaning and feature engineering (including transforming year into `car_age`)

* Exploratory analysis to identify key predictors

* Regression models for price prediction:

      Linear Regression

      Random Forest Regressor

      XGBoost Regressor

* Classification model to categorize cars into age groups:

      Random Forest Classifier

Model comparison based on accuracy, interpretability, and error metrics

## 5. Key Insights
* Converting year into car_age significantly improved model performance across both datasets.

* `car_age`, `engine_size`, and `mileage` consistently emerged as the strongest predictors of price.

* The classification project revealed clear age‑based market patterns, complementing the regression analysis.

* Having two datasets strengthened the robustness of insights and demonstrated adaptability across different market samples.

## 6. Model Performance
* Random Forest Regressor delivered the most balanced and reliable price predictions.

* XGBoost performed competitively but required more tuning.

* Linear Regression provided interpretability but struggled with non‑linear relationships.

* Random Forest Classifier achieved strong accuracy in distinguishing young vs. old vehicles.

## 7. Limitations & Next Steps
The datasets lack geographic information, which limits the ability to model regional price differences. Incorporating dealership region could significantly improve prediction accuracy. Additional features such as brand reputation or market demand could further refine the models.

Future work may explore:

* Gradient boosting with optimized hyperparameters

* Price clustering to identify hidden market segments





