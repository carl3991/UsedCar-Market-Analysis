
# Installing required libraries
import pandas as pd
import numpy as np
import time
import streamlit as st
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm

from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

########################################################################################################3333
st.set_page_config(
    page_title= "Used Car Analytics",
    layout = "wide"
)


# Load data
@st.cache_data
def load_data():
# Get data from GitHub
    url = "https://raw.githubusercontent.com/carl3991/UsedCar-Market-Analysis/main/01_Car/used_car_price_analysis.csv"
    return pd.read_csv(url)

df = load_data()

# Data Cleaning 
df['tax'] = df['tax'].fillna(df['tax'].mean())
df.drop_duplicates(inplace=True)

# Remove data entry error where year is 2060
df = df[df['year'] != 2060]

# Create car age feature
current_year = datetime.now().year
df['car_age'] = current_year - df['year']



#########################################################################################################################################33
# Design sidebar
sidebar_title = st.sidebar.title("Used Car Analytics Dashboard")

navigation = st.sidebar.radio("", ["Home","Exploratory Analysis", "Machine Learning", "Prediction Price Interface"])

########################################################################################################################################
if navigation == "Home":

    st.title("Welcome!")
  
    st.image("https://raw.githubusercontent.com/carl3991/UsedCar-Market-Analysis/main/01_Car/dealership.jpg", width = 870)

    st.divider()
    
    st.subheader("Project Description")
    st.markdown("""
This project explores the factors that influence used car prices and presents those findings through friendly interactive visuals. 
Using a dataset of thousands of vehicle listings, the analysis examines how characteristics such as vehicle age, mileage, fuel type, transmission, engine size, fuel efficiency (MPG), and manufacturer can impact market market value.

The dashboard includes a range of visualizations that help users identify pricing patterns and trends within the used car market.
Users can compare average prices across manufacturers, explore relationships between mileage and vehicle value, examine the impact of age on depreciation, and better understand the characteristics associated with higher- and lower-priced vehicles.

In addition to the exploratory analysis, the project includes predictive models that estimate vehicle prices based on key attributes. 
By combining statistical analysis, data visualization, and machine learning, the dashboard provides a practical way to better understand the factors that drive used car pricing.""")
    st.divider()
    
    st.subheader("About Me")
    st.markdown("""
            My name is **Carl Christian Legros**, a Data Science graduate student and aspiring analytics professional with hands-on experience in machine learning, business intelligence, data visualization, and statistical analysis. 
            I specialize in transforming raw, complex data into actionable insights through interactive dashboards, predictive models, and data-driven storytelling.
            My projects have included business analytics, pricing analysis, operational reporting, and performance optimization, allowing me to develop strong technical skills in Python, SQL, Tableau, Power BI, and data analytics methodologies. 
            I am passionate about leveraging data to solve problems, improve decision-making, and create measurable impact across organizations.
            """)    
    st.divider()
    
    st.subheader("Let's Connect")
    st.info(
                    """
                    [**LinkedIn**](https://www.linkedin.com/in/carllegros/)    
                    [**GitHub**](https://github.com/carl3991)
                    """
                )
################################################################################################

elif navigation == "Exploratory Analysis":
    selected_option = st.sidebar.selectbox("Choose Options", ["Data Preview", "Key Performace Indicator", "Business Insight", "Statistical Analysis"])

    if selected_option == "Data Preview":
        st.subheader("Dataset Info")
        st.write(f"Rows: {df.shape[0]:,.0f}")
        st.write(f"Columns: {df.shape[1]}")
        st.divider()

        st.subheader("Data Preview")
        st.dataframe(df.head(10))
  
##############################################################################################33
    elif selected_option == "Key Performace Indicator":
        st.subheader("Executive Summary")

        # kpi objects
        total_price = df['price'].sum()
        avg_price = df['price'].mean()
        min_price = df['price'].min()
        max_price = df['price'].max()

        # create kpi columns
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        # Display KPI's
        kpi1.metric(label = "Total Price", value = f"${total_price:,.0f}")
        kpi2.metric(label = "Average Price", value = f"${avg_price:,.0f}")
        kpi3.metric(label = "Minimum Price", value = f"${min_price:,.0f}")
        kpi4.metric(label = "Maximum Price", value = f"${max_price:,.0f}")
###########################################################################
    elif selected_option == "Business Insight":
        st.title("Data Exploration")
        st.divider()

        st.subheader("Price Trend")
        
                # create line plot object
        line_price = df.groupby("year", as_index=False)["price"].mean()
        
                # plot price trend
        fig_trend = px.line(line_price, x="year", y="price", title="", markers=True)
        
        fig_trend.update_layout(xaxis_title = "Year", yaxis_title = "Average Price ($)", hoverlabel = dict(
                                                    bgcolor = "white",
                                                    font_color = "black" # custom tooltip font and background color
                                                ), hovermode = "x unified")
        
        fig_trend.update_yaxes(tickprefix="$")
        
        fig_trend.update_traces(line = dict(color="red", width=3), hovertemplate = "<b>%{x}</b><br>"+
                                        "Average Price: <b>$%{y:,.2f}</b>" +
                                        "<extra></extra>")
        
        st.plotly_chart(fig_trend, use_container_width=True)

        st.divider()

        st.subheader("Top 10 Expensive Models")

        # Most Expensive models
        expensive_cars = df.groupby("model", as_index=False)['price'].mean()
        expensive_cars = expensive_cars.sort_values(by="price", ascending = False).head(10)

        # Plot most expensive models
        fig_model = px.bar(expensive_cars, x = "price", y = "model", orientation = "h",title ="", color="price", color_continuous_scale=px.colors.sequential.YlOrRd)

        fig_model.update_layout(xaxis_title = "Average Price ($)", yaxis_title = "Car Model", hoverlabel = dict(
                                            bgcolor = "white",
                                            font_color = "black" # custom tooltip font and background color
                                        ), coloraxis_colorbar = dict(
                                            title = "Price ($)"
                                        ))

        fig_model.update_yaxes(# Force yaxis to go ascending
            categoryorder="total ascending")
        fig_model.update_xaxes(tickprefix = "$" )
        
        fig_model.update_traces(marker_line_color= "black", marker_line_width =1.2, hovertemplate = "<b>%{y}</b> <br>"+ 
                                "Price: <b>$%{x:,.02f}</b>" +
                                "<extra></extra>")

        st.plotly_chart(fig_model, use_container_width=True)

        st.divider()

        st.subheader("Most Expensive Car Transmission")
        
                # Most Expensive transmission
        transmission_cars = df.groupby("transmission", as_index=False)['price'].mean()
        transmission_cars = transmission_cars.sort_values(by="price", ascending = False)
        
                # Plot most expensive transmission
        fig_transmission = px.bar(transmission_cars, x = "transmission", y = "price",title ="", color="price", color_continuous_scale=px.colors.sequential.Blues)
        
        fig_transmission.update_layout(yaxis_title = "Average Price ($)", xaxis_title = "Transmission Type", hoverlabel = dict(
                                                    bgcolor = "white",
                                                    font_color = "black" # custom tooltip font and background color
                                                ), coloraxis_colorbar = dict(
                                                    title = "Price ($)"
                                                ))
        
        fig_transmission.update_yaxes( tickprefix="$")                
        fig_transmission.update_traces(marker_line_color= "black", marker_line_width =1.2, hovertemplate = "<b>%{x}</b> <br>"+ 
                                        "Price: <b>$%{y:,.02f}</b>" +
                                        "<extra></extra>")
        
        st.plotly_chart(fig_transmission, use_container_width=True)

        st.divider()
        
        st.subheader("Most Expensive Fuel Type")
                
                        # Most Expensive transmission
        fuel_cars = df.groupby("fuelType", as_index=False)['price'].mean()
        fuel_cars = fuel_cars.sort_values(by="price", ascending = False)
                
                        # Plot most expensive transmission
        fig_fuel = px.bar(fuel_cars, x = "fuelType", y = "price",title ="", color="price", color_continuous_scale=px.colors.sequential.Greens)
                
        fig_fuel.update_layout(yaxis_title = "Average Price ($)", xaxis_title = "Fuel Type", hoverlabel = dict(
                                                            bgcolor = "white",
                                                            font_color = "black" # custom tooltip font and background color
                                                        ), coloraxis_colorbar = dict(
                                                            title = "Price ($)"
                                                        ))
                
        fig_fuel.update_yaxes(tickprefix = "$")                        
        fig_fuel.update_traces(marker_line_color= "black", marker_line_width =1.2, hovertemplate = "<b>%{x}</b> <br>"+ 
                                                "Price: <b>$%{y:,.02f}</b>" +
                                                "<extra></extra>")
                
        st.plotly_chart(fig_fuel, use_container_width=True)

###########################################################################
    elif selected_option == "Statistical Analysis":
        #histogram
        st.subheader("Price Distribution")
        fig_hist = px.histogram(x=df['price'],nbins=50, title="" )
        fig_hist.update_layout(xaxis_title = "Price ($)", yaxis_title = "Frequency", hoverlabel = dict(font_color="black") )
        fig_hist.update_traces(marker_line_color = "black", marker_line_width = 2,
                               hovertemplate = "<b> Price Distribution</b> <br>" +
                               "# of cars: <b> %{y}</b><br>" +
                               "Price Bin: <b> $%{x}</b>")

        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        st.subheader("Correlation Heatmap")

        # Correlation Matrix object with selected features
        corr = df[['price', 'mileage', 'mpg', 'tax','engineSize']].corr()

        #plot correlation matrix
        fig_corr = px.imshow(corr, text_auto= ".2f", color_continuous_scale= "Greys", aspect = "auto")
        fig_corr.update_layout(height=600, width=900, hoverlabel = dict(
            font_color = "black"
        ))
  
        fig_corr.update_traces(hovertemplate = "%{y} vs. %{x} <br>" + 
                               "Correlation: %{z:,.3f}" +
                               "<extra></extra>")

        st.plotly_chart(fig_corr, use_container_width=True)

        st.divider()
                  
        # scatter plot
        st.subheader("Price vs. Mileage by Fuel Type")
        fig_scatter = px.scatter(df, x="mileage", y="price", color="fuelType", custom_data=["fuelType"], title="")

        fig_scatter.update_layout(
            xaxis_title="Mileage",
            yaxis_title="Price ($)", hoverlabel = dict(font_color="black"), legend = dict(
                title = "Fuel Type"
            )
        )

        fig_scatter.update_traces(hovertemplate = "<b> Price vs. Mileage</b><br>" + 
                                  "Price: <b> $%{y:,.0f}</b><br>" + 
                                  "Mileage : <b> %{x:,.0f}</b><br>" + 
                                  "Fuel Type : <b> %{customdata[0]}</b>" +
                                  "<extra> </extra>")
        fig_scatter.add_scatter(mode = "lines", line = dict(color="red", dash = "solid"))

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()
        # boxplot
        st.subheader("Price Distribution by Fuel Type and Transmission")
        fig_box = px.box(df, x = "fuelType", y ="price", color='transmission')
        fig_box.update_layout(xaxis_title = "Fuel Type" , yaxis_title = "Price ($)", legend = dict(
            title = "Transmission",
            font = dict(color="black"),
            title_font = dict(color="black")
        ))


        st.plotly_chart(fig_box, use_container_width=True)
##########################################################################################################
elif navigation == "Machine Learning":
    selected_option2 = st.sidebar.selectbox("Select Options", ["Understanding Models Metrics", "Train Linear Regression Model", "Train Random Forest Model", "Train XGBoost Model", "Model Comparison"])

    # Metric explanation section
    if selected_option2 == "Understanding Models Metrics":
        st.title("Brief Explanation of Model Metric")
        st.divider()

        st.markdown("""
        
 #### R-Squared Score
 Measures how well the model explains differences in vehicle prices. A higher value indicates that the model captures more of the pricing patterns present in the data. For example, an R² score of 0.81 means the model explains approximately 81% of the variation in used car prices.

#### Root Mean Squared Error (RMSE) 
 Shows the typical prediction error in dollars, while placing greater emphasis on larger mistakes. Lower values indicate more accurate predictions.

 #### Mean Absolute Error (MAE) 
 Represents the average difference between the predicted price and the actual price. For example, an MAE of 1,500 dollars means that predictions are typically off by about 1,500 dollars.

#### Mean Absolute Percentage Error (MAPE) 
 Measures prediction error as a percentage. For example, a MAPE of 5% means predictions are typically within 5% of the vehicle's actual price.

#### Training Time
 Measures how long the model takes to learn from the data.""")
###################################################################################################################################################################################################    
    
    # Linear model section
    elif selected_option2 == "Train Linear Regression Model":
        st.subheader("Linear Regression Summary")

        st.markdown(""" This model estimates used car prices based on factors such as mileage, vehicle age, engine size, fuel efficiency, transmission type, and fuel type. 
        Historical vehicle listings were used to identify pricing patterns and relationships between these characteristics and market value.

To improve prediction accuracy, the model was trained using a transformed version of vehicle prices that reduces the impact of extreme values.
Once predictions were generated, they were converted back into dollar amounts for easier interpretation.

The model provides a straightforward way to understand how different vehicle characteristics influence price and serves as a useful baseline for comparing the performance of more advanced machine learning models. """)
         # Features and target

        X = df[["mileage", "tax", "mpg", "engineSize",
                            "transmission", "fuelType", "car_age"]]
        df['log_price'] = np.log1p(df['price'])    
        y = df["log_price"]
            
                    # One-hot encode categorical variables
        X = pd.get_dummies(
                        X,
                        columns=["transmission", "fuelType"],
                        drop_first=True
                    )
            
                    # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42
                    )
        # Linear regression start time
        lr_start = time.time()
            
                    # Initialize model
        lr = LinearRegression()
            
                    # Fit model
        lr.fit(X_train, y_train)

        # end time 
        lr_end = time.time()
        # training time
        lr_training_time = lr_end - lr_start

        # Store lr model
        st.session_state["lr"] = lr

                    # Predictions
        y_pred_log = lr.predict(X_test)

        # Revert back to actual prices
        y_test_actual = np.expm1(y_test)
        y_pred_actual = np.expm1(y_pred_log)
            
                    # Evaluation
        rmse_lr = mean_squared_error(y_test_actual, y_pred_actual)**0.5
        r2_lr = r2_score(y_test, y_pred_log)
        mae_lr = mean_absolute_error(y_test_actual, y_pred_actual)
        mape_lr = mean_absolute_percentage_error(y_test_actual, y_pred_actual) *100

         # Store lr metrics
        st.session_state["mae_lr"] = mae_lr
        st.session_state["mape_lr"] = mape_lr
        st.session_state["r2_lr"] = r2_lr
        st.session_state["rmse_lr"] = rmse_lr
        st.session_state["lr_training_time"] = lr_training_time

                    # Display metrics
        m5, m6, m7, m8, m9 = st.columns(5)
        m5 = f"{r2_lr:.2f}"
        m6 = f"${rmse_lr:,.2f}"
        m7 = f"${mae_lr:,.2f}"
        m8 = f"{mape_lr:,.2f}%"
        m9 = f"{lr_training_time:,.2f} secs."
            
        st.metric("R-squared Score (with log transformation)", m5)
        st.metric("Root Mean Squared Error", m6)
        st.metric("Mean Absolute Error", m7)
        st.metric("Mean Absolute Percent Error", m8)
        st.metric("Training Time", m9)

        st.divider()
        st.subheader("Coefficient Table")

        # create OLS table summary
        X_ols = sm.add_constant(X)
        ols_model = sm.OLS(y, X_ols).fit()

        coef_df = pd.DataFrame({
        "Feature": ols_model.params.index,
        "Coefficient": ols_model.params.values,
        "P-Value": ols_model.pvalues.values.round(3)
            })

        st.dataframe(coef_df, use_container_width=True)

        st.divider()

                # Actual vs predicted plot

        st.subheader("Actual vs Predicted Prices")

        actual_pred_df = pd.DataFrame({
    "Actual Price": y_test_actual,
    "Predicted Price": y_pred_actual
})

        fig_actual_pred = px.scatter(
    actual_pred_df,
    x="Actual Price",
    y="Predicted Price",
    title=""
)

# Perfect prediction line
        min_val = min(actual_pred_df.min())
        max_val = max(actual_pred_df.max())

        fig_actual_pred.add_scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",
    name="Perfect Prediction",
    line=dict(color="red", dash="dash")
)

        fig_actual_pred.update_layout(
    xaxis_title="Actual Price ($)",
    yaxis_title="Predicted Price ($)",
    hoverlabel=dict(font_color="black")
)

        fig_actual_pred.update_traces(
    hovertemplate=
    "<b>Actual vs Predicted</b><br>" +
    "Actual: <b>$%{x:,.0f}</b><br>" +
    "Predicted: <b>$%{y:,.0f}</b>" +
    "<extra></extra>"
)

        st.plotly_chart(fig_actual_pred, use_container_width=True)

        st.divider()
        st.subheader("Residuals - Histogram")

        #### Plot residuals histogram

        # create residual object
        residuals = y_test_actual - y_pred_actual

        fig_resid = px.histogram(residuals, nbins=30, title="")
        fig_resid.update_layout(xaxis_title = "Residual", yaxis_title="Count")
        fig_resid.update_traces(marker_color = "lime", marker_line_color = "black", marker_line_width = 1.2)

        st.plotly_chart(fig_resid, use_container_width=True)
        st.markdown("""This is an approximately bell shaped histogram centered around zero. This suggest that the residuals are stable.""")

        st.divider()
        st.subheader("Q-Q Plot of Linear Regression Residuals")

        #### Plot QQ 
        import plotly.graph_objects as go
        
        # caluculate QQ Data
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

        qq = go.Figure()
        qq.add_trace(go.Scatter(x = osm, y = osr, mode="markers", name="Residuals"))
        #reference line
        qq.add_trace(go.Scatter(x = osm, y = slope * osm + intercept, mode="lines", name = "Normal Line", line = dict(color="red")))
        qq.update_layout(title ="", xaxis_title = "Theoretical Quantiles", yaxis_title = "Sample Quantiles" )

        st.plotly_chart(qq, use_container_width=True)

        st.metric("QQ Correlation", f"{r:.4f}")

        st.markdown(""" 
            - Most of the points closely follow the red reference line
- No major S-shape pattern

- No severe skewness visible

- **The normality assumption appears reasonably satisfied.**

                """)






    elif selected_option2 == "Train Random Forest Model":
        st.subheader("Random Forest Regressor")
        st.divider()
        st.markdown("""
         This model (Random Forest) was developed to estimate used car prices using the same key characteristics.
         Random Forest combines the results of many decision trees to make a prediction. 
         One major strenght of this model is that it can identify complex relationships between vehicle characteristics and market value, allowing it to capture pricing trends that may be difficult for simpler models to detect, often leading to more accurate and reliable price estimates.


                    """)
                    # Features and target
        X = df[["mileage", "tax", "mpg", "engineSize",
                    "transmission", "fuelType", "car_age"]]
    
        y = df["price"]
    
            # One-hot encode categorical variables
        X = pd.get_dummies(
                X,
                columns=["transmission", "fuelType"],
                drop_first=True
            )
    
            # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

        rf_start = time.time()
            # Initialize model
        rf = RandomForestRegressor(n_estimators=200,random_state=42)
    
            # Fit model
        rf.fit(X_train, y_train)

        rf_end = time.time()
        rf_training_time = rf_end - rf_start

        # Store model
        st.session_state["rf"] = rf
    
            # Predictions
        y_pred_rf = rf.predict(X_test)
    
            # Evaluation
        rmse_rf = mean_squared_error(y_test, y_pred_rf)**0.5
        r2_rf = r2_score(y_test, y_pred_rf)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf) * 100

        # Store rf metrics
        st.session_state["mae_rf"] = mae_rf
        st.session_state["mape_rf"] = mape_rf
        st.session_state["r2_rf"] = r2_rf
        st.session_state["rmse_rf"] = rmse_rf
        st.session_state["rf_training_time"] = rf_training_time
    
            # Display metrics
        r1, r2, r3, r4, r5 = st.columns(5)
        r1 = f"{r2_rf:.2f}"
        r2 = f"${rmse_rf:,.2f}"
        r3 = f"${mae_rf:,.2f}"
        r4 = f"{mape_rf:,.2f}%"
        r5 = f"{rf_training_time:,.2f} sec."

    
        st.metric("R-squared Score", r1)
        st.metric("Root Mean Squared Error", r2)
        st.metric("Mean Absolute Error", r3)
        st.metric("Mean Absolute Percent Error", r4)
        st.metric("Random Forest Training Time", r5)
    
            # Importance creation for XGBoost model
        importances = rf.feature_importances_
        importances_rf = pd.DataFrame({
                "Features": X.columns,
                "Importance": importances
            })
        importances_rf = importances_rf.sort_values(by="Importance", ascending=True)
    
        feat_imp_rf = px.bar(importances_rf, x = "Importance", y="Features", title ="Feature Importance - Random Forest",
                                   color = "Importance", color_continuous_scale= px.colors.sequential.Blues)
        feat_imp_rf.update_layout(title_x = 0.4
                    ),
                    hoverlabel= dict(bgcolor="white", font_color="black")) # custom tooltip 
        feat_imp_rf.update_traces(marker_line_color = "black", marker_line_width = 1.2, 
                                       hovertemplate = "<b>%{y}</b><br>" +
                                       "Importance Score: <b>%{x:,.2f}</b>")
    
    
        st.plotly_chart(feat_imp_rf, use_container_width=True)    

    # Xgboost result page
    elif selected_option2 == "Train XGBoost Model":

        st.subheader("XGBoost Regressor")
        st.divider()
        st.markdown("""


Similar to the other models , this model (**XGBoost**) was developed to estimate used car prices using all vehicle characteristics.

XGBoost improves its predictions through a series of learning steps. Rather than making a single prediction, it repeatedly analyzes previous mistakes and adjusts its approach to improve accuracy. 
Which means the model can identify complex pricing patterns and interactions between vehicle characteristics that may not be obvious through traditional analysis.

Because of its ability to learn from errors, XGBoost often produces highly accurate price estimates and is widely used for predictive modeling tasks.
            """)
                # Features and target
        X = df[["mileage", "tax", "mpg", "engineSize",
                "transmission", "fuelType", "car_age"]]

        y = df["price"]

        # One-hot encode categorical variables
        X = pd.get_dummies(
            X,
            columns=["transmission", "fuelType"],
            drop_first=True
        )

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        xgb_start = time.time()

        # Initialize model
        xgb = XGBRegressor(n_estimators=300, max_depth = 6, learning_rate = 0.05, subsample = 0.8, min_child_weight = 3,
                           colsample_bytree = 0.8, random_state=42)

        # Fit model
        xgb.fit(X_train, y_train)

        xgb_end = time.time()
        xgb_training_time = xgb_end - xgb_start

        # Store model
        st.session_state["xgb"] = xgb

        # Predictions
        y_pred_xgb = xgb.predict(X_test)

        # Evaluation metrics
        rmse_xgb = mean_squared_error(y_test, y_pred_xgb)**0.5
        r2_xgb = r2_score(y_test, y_pred_xgb)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb) * 100

        # Display metrics
        x1, x2, x3, x4, x5 = st.columns(5)
        x1 = f"{r2_xgb:.2f}"
        x2 = f"${rmse_xgb:,.2f}"
        x3 = f"${mae_xgb:,.2f}"
        x4 = f"{mape_xgb:,.2f}%"
        x5 = f"{xgb_training_time:,.2f} sec."

        st.metric("R-squared Score", x1)
        st.metric("Root Mean Squared Error", x2)
        st.metric("Mean Absolute Error", x3)
        st.metric("Mean Absolute Percent Error", x4)
        st.metric("XGBoost Training Time", x5)

         # Store xgb metrics
        st.session_state["mae_xgb"] = mae_xgb
        st.session_state["mape_xgb"] = mape_xgb
        st.session_state["r2_xgb"] = r2_xgb
        st.session_state["rmse_xgb"] = rmse_xgb
        st.session_state["xgb_training_time"] = xgb_training_time

        # Importance creation for XGBoost model
        importances = xgb.feature_importances_
        importances_xgb = pd.DataFrame({
            "Features": X.columns,
            "Importance": importances
        })
        importances_xgb = importances_xgb.sort_values(by="Importance", ascending=True)

        feat_imp_xgb = px.bar(importances_xgb, x = "Importance", y="Features", title ="Feature Importance - XGBoost",
                               color = "Importance", color_continuous_scale= px.colors.sequential.Greens)
        feat_imp_xgb.update_layout(title_x = 0.4, coloraxis_colorbar=dict(# for the color legend custom
                    title_font=dict(color="black"),
                    tickfont=dict(color="black")
                ),
                hoverlabel= dict(bgcolor="white", font_color="black")) # custom tooltip 
        feat_imp_xgb.update_traces(marker_line_color = "black", marker_line_width = 1.2, 
                                   hovertemplate = "<b>%{y}</b><br>" +
                                   "Importance Score: <b>%{x:,.2f}</b>")

        
        st.plotly_chart(feat_imp_xgb, use_container_width=True)

    # Model comparison table
    elif selected_option2 == "Model Comparison":

        # Create training function for each model
            
        st.subheader("Model Summary Table")

        if st.button("Compare Models"):
            if all(st.session_state.get(k) is not None for k in ['lr', 'rf', 'xgb']):

                st.session_state["comparison"] = pd.DataFrame({
            "Metric": ["R-squared", "RMSE", "MAE", "MAPE (%)", "Training Time (sec)"],
            "Linear Regression" : [st.session_state.get("r2_lr"), st.session_state.get("rmse_lr"), st.session_state.get("mae_lr"), st.session_state.get("mape_lr"),
                                st.session_state.get("lr_training_time")],
            "Random Forest": [st.session_state.get("r2_rf"), st.session_state.get("rmse_rf"), st.session_state.get("mae_rf"), st.session_state.get("mape_rf"), 
                                                st.session_state.get("rf_training_time")],
            "XGBoost": [st.session_state.get("r2_xgb"), st.session_state.get("rmse_xgb"), st.session_state.get("mae_xgb"), 
                                            st.session_state.get("mape_xgb"), st.session_state.get("xgb_training_time")]
        })
        
        else:
            st.warning("Please train all three models first before comparison.")

        if "comparison" in st.session_state:
            st.dataframe(st.session_state["comparison"], hide_index=True)


###################################################################################################################
elif navigation == "Prediction Price Interface":

   

    @st.cache_resource
    def train_model():

# Create car age
        current_year = datetime.now().year

        df_ml = df.copy()
        df_ml["car_age"] = current_year - df_ml["year"]

        # Features and target
        X = df_ml[
            [
                "model",
                "transmission",
                "fuelType",
                "car_age",
                "mileage",
                "tax",
                "mpg",
                "engineSize"
            ]
        ]

        y = df_ml["price"]

        # Categorical columns
        categorical_features = [
            "model",
            "transmission",
            "fuelType"
        ]

        # Pipeline preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features
                )
            ],
            remainder="passthrough"
        )

        # Train xgboost model

        xgb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=100,
                random_state=42
            ))
        ])

        # fit xgboost
        xgb_pipeline.fit(X, y)

        return xgb_pipeline

    xgb_model = train_model()


    st.subheader("Price Estimation")
    st.divider()

    st.info(
        "Price predictions are generated using an XGBoost model "
        "with 92% explanatory power (R-squared = 0.92, Average Error = 7.94%).")


    st.markdown("""##### Fill out the form to generate a price based of vehicle characteristics.""")
    with st.form("prediction_form"):

        model = st.selectbox(
        "Vehicle Model",
        sorted(df["model"].unique())
        )
        
        year = st.selectbox("Year",sorted(df["year"].unique(), reverse=True))
        
        fuel = st.selectbox("Fuel Type",sorted(df["fuelType"].unique()) )
        
        transmission = st.selectbox("Transmission",sorted(df["transmission"].unique()))

        mileage = st.number_input("Mileage",min_value = 0 , value = 50000, step = 1000)

        engine_size = st.slider("Engine Size (L)",min_value=0.0 , value=8.0, step=0.1)
        
        tax = st.slider("Tax", min_value=0.0 ,value=150.0, step = 0.01)
        
        mpg = st.slider("Miles Per Gallon",min_value= 0, max_value = 60,  value=50 , step = 1)
                
        submitted = st.form_submit_button("Get Price Estimate")


        if submitted:

            car_age = current_year - year
            input_data = pd.DataFrame({
    "model": [model],
    "transmission": [transmission],
    "fuelType": [fuel],
    "car_age": [car_age],
    "mileage": [mileage],
    "tax": [tax],
    "mpg": [mpg],
    "engineSize": [engine_size]
    })

            prediction = xgb_model.predict(input_data)[0]

            st.markdown("""**Based on your input, a used car estimated price is** """)
            st.metric(label = "", value = f"${prediction:,.2f}")












   
