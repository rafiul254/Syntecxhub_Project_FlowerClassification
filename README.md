# 🌸 Iris Flower Classifier — Web Application

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-99%25-22c55e?style=flat)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)
![Internship](https://img.shields.io/badge/Syntecxhub-ML%20Internship%20Week%202-purple?style=flat)

A full-stack machine learning web application that classifies Iris flowers into 3 species using 4 trained ML models simultaneously — built with Flask, scikit-learn, Chart.js, and vanilla JavaScript.

**[Features](#features) · [Installation](#installation) · [How It Works](#how-it-works) · [Models](#ml-models-and-accuracy) · [API](#api-endpoints) · [Project Structure](#project-structure)**

---

## About

This project is part of the **Syntecxhub ML Internship (Week 2)**. The goal is to build a complete end-to-end machine learning pipeline — from raw data exploration all the way to a fully interactive, production-style web application — using the classic **Iris dataset** introduced by botanist Ronald Fisher in 1936.

Users input 4 flower measurements (sepal length, sepal width, petal length, petal width) using interactive sliders. The app instantly classifies the flower as one of 3 Iris species — **Setosa**, **Versicolor**, or **Virginica** — running 4 different ML models in parallel and showing a side-by-side comparison of all results.

---

## Features

| Feature | Description |
|---|---|
| 🤖 **4 ML Models** | Logistic Regression, Decision Tree, Random Forest, SVM — run in parallel |
| 🌸 **Real Flower Photos** | Actual species photos displayed on idle screen and after prediction |
| 📊 **EDA Dashboard** | 6 interactive Chart.js charts — scatter, radar, bar, doughnut |
| 📈 **Probability Bars** | Per-species confidence scores shown for all 4 models |
| 🗳️ **Model Voting** | Shows how many models agree — e.g. "4/4 models agree" |
| 🕓 **Prediction History** | Stores last 20 predictions in session memory |
| 📤 **CSV Export** | Download all predictions as a timestamped `.csv` file |
| 🎨 **Dark Theme UI** | Fully responsive dark interface with smooth animations |
| ⚡ **REST API** | Clean Flask endpoints — `/predict`, `/history`, `/export`, `/health` |

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

This trains all 4 models and saves them as `.pkl` files inside `outputs/models/`.

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
Flask Backend  →  POST /predict
         ↓
4 ML Models run in parallel
  ├── Random Forest   (primary)
  ├── SVM
  ├── Logistic Regression
  └── Decision Tree
         ↓
Vote count calculated  →  final = Random Forest
         ↓
JSON response → JavaScript renders:
  result card + real flower photo +
  probability bars + history panel
```

---

## ML Models and Accuracy

All models trained on the Iris dataset — 150 samples, 80/20 stratified train-test split.

| Model | Test Accuracy | CV Accuracy (5-fold) | Role |
|---|---|---|---|
| 🌲 Random Forest | ~99% | ~96% | Primary — final prediction |
| ⚡ SVM (RBF kernel) | ~98% | ~98% | Best CV score |
| 📈 Logistic Regression | ~97% | ~97% | Fast, interpretable |
| 🌳 Decision Tree | ~95% | ~96% | Fully explainable rules |

The final prediction is always taken from Random Forest. All 4 model results are displayed side-by-side for comparison.

---

## Dataset

- **Name:** Iris Dataset (Ronald Fisher, 1936)
- **Samples:** 150 total — 50 per species
- **Features:** 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
- **Classes:** Setosa, Versicolor, Virginica
- **Source:** `sklearn.datasets.load_iris()`

### Feature Importance (Random Forest)

| Feature | Importance | Notes |
|---|---|---|
| Petal Length | ~47% | Most discriminative |
| Petal Width | ~38% | Second most important |
| Sepal Length | ~11% | Moderate contribution |
| Sepal Width | ~4% | Least useful |

> Petal features alone can classify ~90%+ of samples correctly.

---

## Project Structure

```
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
| `GET` | `/charts` | EDA dashboard |
| `POST` | `/predict` | Run all 4 models, return JSON |
| `GET` | `/history` | Last 20 predictions as JSON |
| `POST` | `/history/clear` | Clear prediction history |
| `GET` | `/export` | Download predictions as CSV |
| `GET` | `/health` | Server health check |

### Sample Request

```json
POST /predict
{
  "sepal_length": 5.1,
  "sepal_width":  3.5,
  "petal_length": 1.4,
  "petal_width":  0.2
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
| Backend | Python 3.10+, Flask 3.0 |
| ML | scikit-learn — LR, DT, Random Forest, SVM |
| Data | NumPy, joblib |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js 4.4 |
| Fonts | Google Fonts — Sora, DM Sans, JetBrains Mono |

---

## Image Credits

Flower photos sourced from Wikimedia Commons under Creative Commons licenses:

- **Iris Setosa** — Photo by Radomil, CC BY-SA 3.0
- **Iris Versicolor** — Photo by Dlanglois, CC BY-SA 3.0
- **Iris Virginica** — Photo by Frank Mayfield, CC BY-SA 2.0

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
