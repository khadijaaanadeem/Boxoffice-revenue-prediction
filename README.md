

# 🎬 Box Office Revenue Prediction

This project builds a **machine learning model** to predict movie box office revenue based on historical movie data, including metadata, budget, runtime, and other relevant features.

The system demonstrates the use of **data preprocessing, feature engineering, and regression modeling** to forecast revenue.

---

## ✨ Features

✅ **Data Cleaning and Preparation:**
Handles missing values and outliers in the dataset to improve prediction accuracy.

✅ **Feature Engineering:**
Transforms categorical variables (e.g., genre, MPAA ratings) and numeric attributes into machine-readable formats.

✅ **Regression Modeling:**
Trains a Random Forest Regressor to predict movie revenue.

✅ **Model Evaluation:**
Evaluates performance using Mean Squared Error (MSE) and R² metrics.

✅ **Prediction Pipeline:**
Provides a workflow for generating revenue predictions on new movie data.

---

## 🛠️ Technologies Used

* Python
* pandas & NumPy
* scikit-learn

  * RandomForestRegressor
  * LabelEncoder
  * train\_test\_split
* Matplotlib
* Seaborn

---

## 📂 Dataset

The dataset includes:

* **Movie metadata:** title, genre, MPAA rating, release year
* **Numerical features:** budget, runtime, and more
* **Target variable:** box office revenue

---

## 🚀 How to Use

1. **Load the dataset**
   Import the dataset into a pandas DataFrame and inspect data quality.

2. **Preprocess Data**

   * Drop or fill missing values
   * Encode categorical features
   * Scale numeric columns if needed

3. **Train the Model**
   Split data into training and testing sets and fit a Random Forest Regressor.

4. **Evaluate Model Performance**
   Compute prediction error metrics to assess model accuracy.

5. **Generate Predictions**
   Use the trained model to predict box office revenue for unseen data.

---

## 💡 Example

```python
# Load and clean data
df = pd.read_csv('movies.csv')
df.drop("budget", axis=1, inplace=True)
for col in ['MPAA', 'genres']:
    df[col] = df[col].fillna(df[col].mode()[0])
df.dropna(inplace=True)

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['MPAA_encoded'] = encoder.fit_transform(df['MPAA'])

# Train/test split
from sklearn.model_selection import train_test_split
X = df[['runtime', 'MPAA_encoded']]
y = df['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

---

## 📈 Results

The model achieved an R² score of approximately **(insert score here)**, demonstrating its ability to capture revenue trends based on input features.

---

## 🎯 Purpose

This project showcases **practical regression modeling techniques** for predicting continuous variables in real-world business contexts such as movie revenue forecasting.

---

## 📈 Future Improvements

* Incorporate additional features such as cast popularity, marketing budget, and release seasonality.
* Experiment with other regression algorithms (XGBoost, Gradient Boosting).
* Deploy as a web application for user-friendly predictions.

---

## 🌟 Acknowledgments

This project was inspired by educational resources and aims to demonstrate machine learning workflows end to end.

---

If you like, I can help you **customize this README further** or adapt it to fit the exact columns and methods you used!
# Boxoffice-revenue-prediction
