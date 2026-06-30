# Car Price Predictor

An end-to-end Machine Learning project that predicts the selling price of a  car based on real-world features like brand, year, mileage, fuel type, engine specs, and more — powered by **XGBoost** and deployed as an interactive **Streamlit** web app.

---

## Demo

> Fill in the car details → Click **Estimate Price** → Get an instant price in ₹ Lakh / ₹ Crore

---

## Model Performance

| Metric | Score |
|---|---|
| R² Score | 0.895 |
| Adjusted R² | 0.890 |
| MAE | ₹ 1,42,124 |
| RMSE | ₹ 2,78,407 |
| Cross-Val R² (5-fold) | **0.922** |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Model | XGBoost Regressor |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Encoding | Scikit-learn LabelEncoder |
| Web App | Streamlit |

---

## Project Structure

```
Used_Car_Price_Prediction/
│
├── app.py               # Data cleaning, feature engineering, model training & saving
├── streamlit_app.py     # Streamlit web app for price prediction
├── car data.csv         # Raw dataset (2000+ used car listings)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

> **Note:** `model_bundle.pkl` is generated locally by running `app.py`. It is not included in the repository.

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/used-car-price-prediction.git
cd used-car-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python app.py
```
This will process the dataset and save `model_bundle.pkl` in the project folder.

### 4. Launch the web app
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`

---

## Features Used

- Car Brand (Make)
- Car Model
- Year of Manufacture
- Kilometres Driven
- Fuel Type (Petrol / Diesel / CNG / Electric / Hybrid)
- Transmission (Manual / Automatic)
- Drivetrain (FWD / RWD / AWD)
- Owner History
- Seller Type
- Car Color
- Engine Size (CC)
- Max Power (bhp)
- Max Torque (Nm)
- Car Dimensions (Length, Width, Height)
- Seating Capacity
- Fuel Tank Capacity
- City / Location

---

## requirements.txt

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
streamlit
```




