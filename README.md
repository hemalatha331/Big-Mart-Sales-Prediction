# Big Mart Sales Prediction

![Big Mart Sales](https://github.com/hemalatha331/Big-Mart-Sales-Prediction/blob/main/Images/big1.png)

## 📌 Project Overview
In today's retail industry, shopping malls and Big Marts keep track of sales data for each individual item to predict future demand and optimize inventory management. These data stores contain vast amounts of customer transactions and product attributes stored in a data warehouse. 

By applying data mining techniques, we can identify frequent patterns and anomalies, allowing us to predict future sales volumes using various machine learning techniques. This project aims to develop a predictive model to assist retailers like Big Mart in making data-driven decisions.

## 🚀 Tech Stack
- **Programming Language:** Python 🐍
- **Machine Learning Libraries:** Scikit-learn, Pandas, NumPy, XGBoost, Matplotlib
- **Model Deployment:** Flask 🌐

---

## 📊 Model Selection & Training

The project explores different machine learning algorithms to enhance sales prediction accuracy:

### ✅ Linear Regression
Used as a baseline model to assess the fundamental relationship between features and sales.

### ✅ Random Forest
An ensemble learning method that improves predictive accuracy by reducing overfitting.

### ✅ XGBoost
A powerful gradient boosting algorithm applied to further enhance the model's performance.

![Big Mart Model Training](https://github.com/hemalatha331/Big-Mart-Sales-Prediction/blob/main/Images/big2.png)

---

## 🌍 Flask Application Setup
The trained sales prediction model is deployed using a **Flask web application**, enabling real-time interaction:

- The Flask app serves as the interface for receiving input features.
- It processes the request and predicts the expected sales volume.
- The result is returned to the user dynamically.

### 🔧 Steps to Run the Flask App:
1. Clone this repository:
   ```sh
   git clone https://github.com/hemalatha331/Big-Mart-Sales-Prediction.git
