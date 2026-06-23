# 🏠 House Price Prediction using Machine Learning & Streamlit

## 📌 Project Overview

This project predicts house prices using Machine Learning Regression algorithms and provides an interactive web interface using Streamlit.

The application allows users to:

- Analyze housing data
- Visualize trends and correlations
- Compare Linear, Ridge, and Lasso Regression models
- Predict house prices using custom inputs

This project demonstrates a complete Machine Learning workflow from data preprocessing to deployment using Streamlit.

---

# 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

---

# 📂 Project Structure

House_Price_Prediction/

├── app.py

├── Housing.csv

├── requirements.txt

└── README.md

---

# 📊 Dataset

The project uses the Housing Dataset.

### Features

| Feature | Description |
|----------|-------------|
| area | House area in square feet |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of floors |
| mainroad | Access to main road |
| guestroom | Availability of guest room |
| basement | Availability of basement |
| hotwaterheating | Hot water heating |
| airconditioning | Air conditioning |
| parking | Parking spaces |
| prefarea | Preferred area |
| furnishingstatus | Furnishing condition |
| price | Target variable |

---

# ⚙️ Machine Learning Workflow

## Step 1: Data Loading

The housing dataset is loaded using Pandas.

```python
df = pd.read_csv("Housing.csv")
```

## Step 2: Data Exploration

- Dataset shape
- Missing values
- Statistical summary
- Data information

## Step 3: Data Preprocessing

### Convert Yes/No Columns

```python
yes → 1
no → 0
```

Columns:

- mainroad
- guestroom
- basement
- hotwaterheating
- airconditioning
- prefarea

### One Hot Encoding

```python
furnishingstatus
```

Converted into:

```python
furnishingstatus_semi-furnished
furnishingstatus_unfurnished
```

## Step 4: Feature Scaling

```python
StandardScaler()
```

Used to normalize input features before training.

## Step 5: Train-Test Split

```python
train_test_split()
```

- 80% Training Data
- 20% Testing Data

---

# 🤖 Models Used

## 1. Linear Regression

```python
LinearRegression()
```

Used as the baseline model.

### Advantages

- Simple
- Fast
- Easy to interpret

---

## 2. Ridge Regression

```python
Ridge(alpha=1.0)
```

Uses L2 Regularization.

### Advantages

- Reduces overfitting
- Handles multicollinearity

---

## 3. Lasso Regression

```python
Lasso(alpha=0.1)
```

Uses L1 Regularization.

### Advantages

- Feature selection
- Removes less important features

---

# 📈 Evaluation Metrics

Models are evaluated using:

## Mean Absolute Error (MAE)

```text
Average absolute prediction error
```

## Root Mean Squared Error (RMSE)

```text
Square root of Mean Squared Error
```

## R² Score

```text
Explains how much variance is captured
```

### Example Results

| Model | MAE | RMSE | R² Score |
|---------|---------|---------|---------|
| Linear Regression | 970043 | 1324507 | 0.65 |
| Ridge Regression | Similar | Similar | Similar |
| Lasso Regression | Similar | Similar | Similar |

---

# 📊 Visualizations

The project generates:

### Correlation Heatmap

Shows relationship between features.

### Price Distribution

Shows how house prices are distributed.

### Area vs Price Plot

Shows relationship between area and price.

### Residual Plot

Used to evaluate prediction errors.

### Actual vs Predicted Plot

Compares actual and predicted prices.

---

# 🌐 Streamlit Application

The Streamlit interface provides:

## Home Page

- Project Overview
- Dataset Information

## Data Analysis Page

- Dataset Preview
- Correlation Heatmap
- Price Distribution

## Model Performance Page

Displays:

- Linear Regression Metrics
- Ridge Regression Metrics
- Lasso Regression Metrics

## House Price Prediction Page

User enters:

- Area
- Bedrooms
- Bathrooms
- Stories
- Main Road
- Guest Room
- Basement
- Hot Water Heating
- Air Conditioning
- Parking
- Preferred Area
- Furnishing Status

The application predicts:

```text
Estimated House Price
```

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/house-price-prediction.git

cd house-price-prediction
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 requirements.txt

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

After running:

```text
Local URL: http://localhost:8501
```

Open the URL in your browser.

---

# 🧪 Sample Prediction

## Input

```text
Area = 7000

Bedrooms = 4

Bathrooms = 2

Stories = 2

Main Road = Yes

Guest Room = No

Basement = Yes

Hot Water Heating = No

Air Conditioning = Yes

Parking = 2

Preferred Area = Yes

Furnishing Status = Furnished
```

## Output

```text
Predicted House Price

₹ 8,456,231
```

---

# 🎯 Future Enhancements

- Hyperparameter Tuning
- Cross Validation
- Random Forest Regression
- XGBoost Regression
- Model Serialization using Pickle
- Streamlit Cloud Deployment
- Feature Importance Dashboard
- Real Estate Recommendation System

---

# 📚 Learning Outcomes

By completing this project, you will learn:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Linear Regression
- Ridge Regression
- Lasso Regression
- Model Evaluation
- Streamlit Development
- End-to-End Machine Learning Workflow

---

# 👩‍💻 Author

Yukktha Srisaila

QA Intern | AI/ML Enthusiast

Skills:
- Python
- Machine Learning
- SQL
- Streamlit
- Playwright Automation
- Data Analysis

---

# 📄 License

This project is developed for educational, learning, and portfolio purposes.
