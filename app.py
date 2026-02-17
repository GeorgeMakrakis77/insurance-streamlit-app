import kagglehub
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# 1. DATA LOADING
# =========================

path = kagglehub.dataset_download("mirichoi0218/insurance")
df = pd.read_csv(f"{path}/insurance.csv")

st.title("Medical Insurance Cost Analysis & Prediction App")

st.write("""
This app performs exploratory data analysis (EDA) on the Medical Cost Personal Dataset 
and builds a Linear Regression model to predict insurance charges.
""")

# =========================
# DATA OVERVIEW
# =========================

st.header("1️⃣ Dataset Overview")

st.subheader("Preview of Data")
st.dataframe(df.head())

st.subheader("Column Descriptions")

st.markdown("""
- **age**: Age of the individual  
- **sex**: Gender (male/female)  
- **bmi**: Body Mass Index  
- **children**: Number of dependents  
- **smoker**: Smoking status  
- **region**: Residential region  
- **charges**: Annual medical insurance cost (target variable)
""")

# Missing / Unusual Value Check
st.subheader("Missing and Unusual Values Check")

missing_values = df.isnull().sum()
blank_values = (df == "").sum()

st.write("Missing values per column:")
st.write(missing_values)

st.write("Blank string values per column:")
st.write(blank_values)

# =========================
# 2. EXPLORATORY DATA ANALYSIS
# =========================

st.header("2️⃣ Exploratory Data Analysis")

st.subheader("Summary Statistics (Numerical Variables)")
st.write(df.describe())

# Distribution of charges
fig1 = px.histogram(df, x="charges", nbins=40,
                    title="Distribution of Insurance Charges")
st.plotly_chart(fig1)

st.write("""
Insurance charges are right-skewed, meaning a small number of individuals 
have very high medical costs.
""")

# Smoker vs charges
fig2 = px.box(df, x="smoker", y="charges",
              color="smoker",
              title="Charges by Smoking Status")
st.plotly_chart(fig2)

# Region vs charges
fig3 = px.box(df, x="region", y="charges",
              color="region",
              title="Charges by Region")
st.plotly_chart(fig3)

# Age vs charges
fig4 = px.scatter(df, x="age", y="charges",
                  trendline="ols",
                  title="Age vs Charges")
st.plotly_chart(fig4)

# BMI vs charges
fig5 = px.scatter(df, x="bmi", y="charges",
                  color="smoker",
                  trendline="ols",
                  title="BMI vs Charges")
st.plotly_chart(fig5)

# Children vs charges
fig6 = px.box(df, x="children", y="charges",
              title="Number of Children vs Charges")
st.plotly_chart(fig6)

# Sex vs charges
fig7 = px.box(df, x="sex", y="charges",
              color="sex",
              title="Sex vs Charges")
st.plotly_chart(fig7)

st.write("""
Key observations:
- Smokers have significantly higher charges.
- Charges increase with age.
- Higher BMI is associated with higher charges, especially for smokers
- Number of children has a moderate effect.
- Sex and region differences are relatively small.
""")

# =========================
# 3. PREDICTIVE MODEL
# =========================

st.header("3️⃣ Linear Regression Model")

# One-hot encoding
df_model = pd.get_dummies(df, drop_first=True)

X = df_model.drop("charges", axis=1)
y = df_model["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("Model Performance")

st.write(f"R² Score: {r2:.3f}")
st.write(f"Mean Absolute Error (MAE): ${mae:,.2f}")

st.markdown("""
**What is R²?**  
R² measures how much of the variation in insurance charges is explained by the model.  
For example, an R² of 0.78 means 78% of the variation is explained.

**What is MAE?**  
MAE (Mean Absolute Error) measures the average dollar difference 
between predicted and actual charges.
""")

st.write("""
Features used:
- Age
- BMI
- Children
- Smoking status
- Sex
- Region

Linear Regression was chosen because:
- Charges are continuous
- It provides interpretable results
- It is a strong baseline model
""")

# =========================
# 4. USER PREDICTION
# =========================

st.header("4️⃣ Predict Your Insurance Charge")

age = st.slider("Age", 18, 65, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", df["region"].unique())

input_data = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0,
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(input_df)[0]

st.success(f"Predicted Insurance Charge: ${prediction:,.2f}")

# =========================
# 5. CONCLUSIONS
# =========================

st.header("5️⃣ Conclusions")

st.write("""
### What factors influence charges the most?

- Smoking status has the largest impact.
- Age and BMI significantly increase charges.
- Children have a moderate effect.
- Region and sex have smaller effects.

### Limitations

- Linear Regression assumes linear relationships.
- No interaction terms were included.
- Real-world insurance pricing uses additional variables.
- The dataset is relatively small.

### Summary        
I performed exploratory data analysis to understand the distribution of insurance charges and how 
demographic and health-related factors influence costs. Based on the findings, I built a Linear Regression 
model using age, BMI, number of children, smoking status, sex, and region, as these variables showed meaningful
relationships with charges. The model explains a substantial portion of the variance
in insurance costs (R² ≈ 0.78), with smoking status, age, and BMI emerging as the strongest predictors.
""")
