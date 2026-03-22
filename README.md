# 🌸 Iris Flower Classifier — Web Application

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-99%25-22c55e?style=flat)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)
![Internship](https://img.shields.io/badge/Syntecxhub-ML%20Internship%20Week%202-purple?style=flat)

A full-stack machine learning web application for classifying Iris flowers into 3 species using 4 trained ML models — built with Flask, scikit-learn, and vanilla JS.

**[Features](#features) · [Installation](#installation) · [How It Works](#how-it-works) · [Models](#ml-models-and-accuracy) · [API](#api-endpoints) · [Project Structure](#project-structure)**

---

## About

This project is built as part of the **Syntecxhub ML Internship (Week 2)**. The goal is to build an end-to-end machine learning pipeline — from data exploration to a fully interactive web application — using the classic **Iris dataset** first introduced by botanist Ronald Fisher in 1936.

The app allows users to input 4 flower measurements (sepal length, sepal width, petal length, petal width) using interactive sliders and instantly classify the flower as one of 3 Iris species — **Setosa**, **Versicolor**, or **Virginica** — using 4 different ML models running simultaneously.

---

## Features

| Feature | Description |
|---|---|
| 🤖 **4 ML Models** | Logistic Regression, Decision Tree, Random Forest, SVM — all run in parallel |
| 📊 **EDA Dashboard** | 6 interactive Chart.js visualizations — scatter plots, radar, bar, doughnut |
| 📈 **Probability Bars** | Confidence scores for each species, per model |
| 🗳️ **Model Voting** | Tracks how many models agree on the prediction |
| 🕓 **Prediction History** | Stores last 20 predictions in memory |
| 📤 **CSV Export** | Download all predictions as a timestamped CSV file |
| 🎨 **Dark Theme UI** | Fully responsive dark-themed interface with smooth animations |
| ⚡ **REST API** | Clean Flask API with `/predict`, `/history`, `/export`, `/health` endpoints |

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/Syntecxhub_Project_FlowerClassification
cd Syntecxhub_Project_FlowerClassification
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Train the models (run once)

```bash
python model/train_model.py
```

This trains all 4 models and saves them as `.pkl` files in `outputs/models/`.

### Step 4 — Start the Flask server

```bash
python app.py
```

### Step 5 — Open in browser

```
http://localhost:5000
```

---

## How It Works

```
User Input (4 measurements via sliders)
        ↓
Flask Backend receives POST /predict
        ↓
All 4 models run in parallel
        ↓
Results compared — vote count calculated
        ↓
Random Forest used as final prediction
        ↓
JSON response sent back to browser
        ↓
JavaScript renders results, charts, history
```

---

## ML Models and Accuracy

All models are trained on the standard Iris dataset with 150 samples, 80/20 train-test split (stratified).

| Model | Test Accuracy | CV Accuracy (5-fold) | Notes |
|---|---|---|---|
| 🌲 Random Forest | ~99% | ~96% | Primary model — highest accuracy |
| ⚡ SVM (RBF kernel) | ~98% | ~98% | Best cross-validation score |
| 📈 Logistic Regression | ~97% | ~97% | Fast and interpretable |
| 🌳 Decision Tree | ~95% | ~96% | Most explainable — if/else rules |

Final prediction is based on Random Forest output. All 4 model results are shown side-by-side for comparison.

---

## Dataset

- **Name:** Iris Dataset (Fisher, 1936)
- **Samples:** 150 (50 per species)
- **Features:** 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
- **Target Classes:** Setosa, Versicolor, Virginica
- **Source:** `sklearn.datasets.load_iris()`

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| Petal Length | ~47% |
| Petal Width | ~38% |
| Sepal Length | ~11% |
| Sepal Width | ~4% |

Petal features are far more discriminative than sepal features.

---

## Project Structure
Syntecxhub_Project_FlowerClassification/
│
├── app.py                        # Flask backend — all routes and API logic
│
├── model/
│   └── train_model.py            # Train 4 models, save .pkl files + metadata
│
├── templates/
│   ├── index.html                # Main classifier page
│   └── charts.html               # EDA dashboard (6 Chart.js visualizations)
│
├── static/
│   ├── css/style.css             # Full dark theme stylesheet
│   ├── js/app.js                 # Slider logic, API calls, result rendering
│   └── images/
│       ├── setosa.jpg            # Real Iris Setosa photo
│       ├── versicolor.jpg        # Real Iris Versicolor photo
│       └── virginica.jpg         # Real Iris Virginica photo
│
├── outputs/
│   └── models/                   # Auto-generated after train_model.py
│       ├── logistic_regression.pkl
│       ├── decision_tree.pkl
│       ├── random_forest.pkl
│       ├── svm.pkl
│       ├── scaler.pkl
│       └── metadata.pkl
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main classifier page |
| `GET` | `/charts` | EDA dashboard page |
| `POST` | `/predict` | Run all 4 models, returns JSON results |
| `GET` | `/history` | Get last 20 predictions as JSON |
| `POST` | `/history/clear` | Clear prediction history |
| `GET` | `/export` | Download predictions as CSV |
| `GET` | `/health` | Server health check |

### Sample Request

```json
POST /predict
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Sample Response

```json
{
  "final_prediction": "setosa",
  "agreement": "4/4 models agree",
  "vote_count": 4,
  "results": {
    "random_forest":       { "species": "setosa", "confidence": 100.0 },
    "logistic_regression": { "species": "setosa", "confidence": 99.8  },
    "decision_tree":       { "species": "setosa", "confidence": 100.0 },
    "svm":                 { "species": "setosa", "confidence": 99.5  }
  }
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn (LR, DT, RF, SVM), joblib, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js 4.4 |
| Fonts | Google Fonts (Sora, DM Sans, JetBrains Mono) |
| Model Storage | joblib `.pkl` files |

---

## Image Credits
Flower photos sourced from Wikimedia Commons under Creative Commons licenses:

Iris Setosa — Photo by Radomil, CC BY-SA 3.0
Iris Versicolor — Photo by Dlanglois, CC BY-SA 3.0
Iris Virginica — Photo by Frank Mayfield, CC BY-SA 2.0

---

## 👤 Author

**Rafiul Islam**
- 🎓 Undergraduate Student, IoT & Robotics — UFTB
- 🏢 Syntecxhub ML Internship — Week 2
- 📧 rafuulislam2004@gmail.com
- 🐙 GitHub:(https://github.com/rafiul254)

---

## 📄 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

<div align="center">
Made with ❤️ as part of Syntecxhub ML Internship ·
</div>
