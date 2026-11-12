# Employee Attrition Prediction API

A FastAPI service that predicts whether an employee is likely to leave the company (attrition), using an XGBoost model trained on the IBM HR Analytics Employee Attrition dataset. It is set up to deploy on [Render](https://render.com).

**Live demo:** ` https://employee-attrition-performancefastapi-1.onrender.com` 

---

## Features

- **REST API**: send one employee's details as JSON to `/predict` and get back `Yes` or `No`.
- **Full ML pipeline in one file**: imputation, scaling, one-hot encoding and the XGBoost classifier are saved together in a single scikit-learn `Pipeline` (`employee_attrition.pkl`). Raw input needs no manual preprocessing.
- **Fixed feature order**: `columns.pkl` stores the training column order, so every request is lined up with the columns the model expects.
- **Auto-generated docs**: Swagger UI at `/docs` and ReDoc at `/redoc`.
- **Exploratory analysis**: `analysis.ipynb` looks at the target balance and the categorical feature distributions.

---

## Tech Stack

| Layer         | Tools                                                         |
|---------------|---------------------------------------------------------------|
| Language      | Python 3.11                                                   |
| API           | FastAPI, Uvicorn                                              |
| ML            | scikit-learn 1.7.2 (Pipeline, ColumnTransformer), XGBoost     |
| Data          | pandas, NumPy                                                 |
| Serialization | joblib                                                        |
| EDA           | Jupyter, Matplotlib, Seaborn                                  |
| Deployment    | Render                                                        |

---

## Project Structure

```
Employee_attritaion_project/
├── app/
│   ├── __init__.py
│   └── app2.py               # FastAPI application (routes + model loading)
├── analysis.ipynb            # Exploratory data analysis
├── employee_attrition.pkl    # Trained pipeline (preprocessing + XGBClassifier)
├── columns.pkl               # Ordered list of input feature names
├── requirements.txt
└── README.md
```

---

## Model

| Item          | Details                                                                                 |
|---------------|-----------------------------------------------------------------------------------------|
| Dataset       | [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset): 1,470 employees, 35 columns |
| Target        | `Attrition` (`Yes` / `No`). The classes are imbalanced: 1,233 No and 237 Yes |
| Preprocessing | `SimpleImputer` → `StandardScaler` for numeric columns, `OneHotEncoder` for categorical columns |
| Classifier    | `XGBClassifier` (binary)                                                                |

> **Note:** scikit-learn is pinned to `1.7.2` because the pickled pipeline was saved with that version. Loading it with a different version can raise errors or give wrong results.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/uday2319/Employee-Attrition-performanceFASTAPI.git
cd Employee-Attrition-performanceFASTAPI
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the server

Run this from the **project root**. The app loads the `.pkl` files by relative path, so it will not find them if you start it from another folder.

```bash
uvicorn app.app2:app --reload
```

The API is now running at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

---

## API Reference

### `GET /`

Health check.

```json
{ "message": "FastAPI running" }
```

### `POST /predict`

Predicts attrition for a single employee. The request body is a JSON object whose keys are the feature names below. Any feature you leave out is sent to the model as missing and filled in by the pipeline's imputer, but you will get the most reliable predictions by sending all of them.

**Example request**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 41,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 1102,
    "Department": "Sales",
    "DistanceFromHome": 1,
    "Education": 2,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EmployeeNumber": 1,
    "EnvironmentSatisfaction": 2,
    "Gender": "Female",
    "HourlyRate": 94,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Sales Executive",
    "JobSatisfaction": 4,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5993,
    "MonthlyRate": 19479,
    "NumCompaniesWorked": 8,
    "Over18": "Y",
    "OverTime": "Yes",
    "PercentSalaryHike": 11,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 1,
    "StandardHours": 80,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 0,
    "WorkLifeBalance": 1,
    "YearsAtCompany": 6,
    "YearsInCurrentRole": 4,
    "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 5
  }'
```

**Response**

```json
{ "Attrition": "Yes" }
```

`Attrition` is `"Yes"` if the model predicts the employee will leave and `"No"` otherwise.

**Python example**

```python
import requests

employee = {"Age": 41, "BusinessTravel": "Travel_Rarely", "Department": "Sales", ...}
r = requests.post("http://127.0.0.1:8000/predict", json=employee)
print(r.json())
```

### Input features

| Feature | Type | Allowed values / range |
|---|---|---|
| `Age` | int | 18–60 |
| `BusinessTravel` | str | `Non-Travel`, `Travel_Rarely`, `Travel_Frequently` |
| `DailyRate` | int | 102–1499 |
| `Department` | str | `Sales`, `Research & Development`, `Human Resources` |
| `DistanceFromHome` | int | 1–29 |
| `Education` | int | 1 Below College · 2 College · 3 Bachelor · 4 Master · 5 Doctor |
| `EducationField` | str | `Life Sciences`, `Medical`, `Marketing`, `Technical Degree`, `Human Resources`, `Other` |
| `EmployeeCount` | int | always `1` |
| `EmployeeNumber` | int | employee ID |
| `EnvironmentSatisfaction` | int | 1 Low – 4 Very High |
| `Gender` | str | `Male`, `Female` |
| `HourlyRate` | int | 30–100 |
| `JobInvolvement` | int | 1 Low – 4 Very High |
| `JobLevel` | int | 1–5 |
| `JobRole` | str | `Sales Executive`, `Research Scientist`, `Laboratory Technician`, `Manufacturing Director`, `Healthcare Representative`, `Manager`, `Sales Representative`, `Research Director`, `Human Resources` |
| `JobSatisfaction` | int | 1 Low – 4 Very High |
| `MaritalStatus` | str | `Single`, `Married`, `Divorced` |
| `MonthlyIncome` | int | 1009–19999 |
| `MonthlyRate` | int | 2094–26999 |
| `NumCompaniesWorked` | int | 0–9 |
| `Over18` | str | always `Y` |
| `OverTime` | str | `Yes`, `No` |
| `PercentSalaryHike` | int | 11–25 |
| `PerformanceRating` | int | 1 Low · 2 Good · 3 Excellent · 4 Outstanding |
| `RelationshipSatisfaction` | int | 1 Low – 4 Very High |
| `StandardHours` | int | always `80` |
| `StockOptionLevel` | int | 0–3 |
| `TotalWorkingYears` | int | 0–40 |
| `TrainingTimesLastYear` | int | 0–6 |
| `WorkLifeBalance` | int | 1 Bad · 2 Good · 3 Better · 4 Best |
| `YearsAtCompany` | int | 0–40 |
| `YearsInCurrentRole` | int | 0–18 |
| `YearsSinceLastPromotion` | int | 0–15 |
| `YearsWithCurrManager` | int | 0–17 |

Ranges are the ones in the training data. Values outside them are accepted, but predictions for them are less reliable.

---

## Deploying to Render

1. Push this repository to GitHub.
2. On Render, choose **New → Web Service** and connect the repository.
3. Use these settings:

| Setting       | Value                                              |
|---------------|----------------------------------------------------|
| Runtime       | Python 3                                           |
| Build command | `pip install -r requirements.txt`                  |
| Start command | `uvicorn app.app2:app --host 0.0.0.0 --port $PORT` |

4. Optionally, pin the Python version by setting the environment variable `PYTHON_VERSION=3.11.9`.

To run with Gunicorn and several Uvicorn workers instead, add `gunicorn` to `requirements.txt` and use this start command:

```bash
gunicorn app.app2:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 2
```

---

## Exploratory Data Analysis

`analysis.ipynb` covers:

- Loading the dataset and viewing summary statistics (`head`, `describe`)
- The target distribution, which is imbalanced: about 16% of employees left
- Count plots for each categorical feature (BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime, and others)

To run it, download the dataset CSV from Kaggle and update the path in the `pd.read_csv(...)` cell.

---

## Possible Improvements

- Add a Pydantic request model so inputs are validated and the example appears in `/docs`
- Return the prediction probability alongside the label
- Add a `/predict/batch` endpoint for scoring several employees at once
- Commit the training script or notebook so the model can be reproduced
- Add tests and a Dockerfile

---

## Author

**Uday**, [GitHub @uday2319](https://github.com/uday2319)

If you find this project useful, consider giving it a ⭐.
