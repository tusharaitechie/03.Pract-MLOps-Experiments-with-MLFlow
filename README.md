# 🚀 MLflow Experiments with Scikit-Learn

A hands-on MLOps project demonstrating how to use **MLflow** for experiment tracking, model logging, artifact management, and hyperparameter tuning with **Scikit-Learn**.

---

# 📌 Project Overview

This repository demonstrates the complete workflow of experiment tracking using **MLflow**.

The project covers:

* Experiment Tracking
* Parameter Logging
* Metric Logging
* Artifact Logging
* Model Logging
* Dataset Logging
* Tags
* Nested Runs
* Hyperparameter Tuning using GridSearchCV

---

# 📂 Project Structure

```text
03.Pract-MLOps-Experiments-with-MLFlow/
│
├── mlruns/                 # MLflow experiment logs
├── src/
│   ├── file1.py            # Basic MLflow Experiment
│   ├── hypertune1.py       # Hyperparameter Tuning using MLflow
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠️ Technologies Used

* Python
* MLflow
* Scikit-Learn
* Pandas
* Matplotlib
* Seaborn

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/tusharaitechie/03.Pract-MLOps-Experiments-with-MLFlow.git
```

Go inside the project

```bash
cd 03.Pract-MLOps-Experiments-with-MLFlow
```

Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Install MLflow

```bash
pip install mlflow
```

Verify installation

```bash
mlflow --version
```

---

# ▶️ Running the Project

Run the basic experiment

```bash
python src/file1.py
```

Run Hyperparameter Tuning

```bash
python src/hypertune1.py
```

---

# 📊 Start MLflow UI

Open a new terminal inside the project directory and run

```bash
mlflow ui
```

By default MLflow starts on

```text
http://127.0.0.1:5000
```

Open the above URL in your browser.

---

# 🌐 Using MLflow Tracking Server

If you want to log experiments through the MLflow Tracking Server, uncomment the following line in your Python script:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

Start the server first

```bash
mlflow ui
```

Then execute your Python script

```bash
python src/file1.py
```

or

```bash
python src/hypertune1.py
```

---

# 📁 MLflow Commands

## Start MLflow UI

```bash
mlflow ui
```

---

## Start UI on Custom Port

```bash
mlflow ui --port 8080
```

Access

```text
http://127.0.0.1:8080
```

---

## Start UI on Custom Host

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

---

## Start MLflow Tracking Server

```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./artifacts \
--host 0.0.0.0 \
--port 5000
```

---

## Check MLflow Version

```bash
mlflow --version
```

---

## Display Help

```bash
mlflow --help
```

---

# 📈 MLflow Features Used

✔ Experiment Tracking

✔ Parameter Logging

✔ Metric Logging

✔ Artifact Logging

✔ Source Code Logging

✔ Model Logging

✔ Dataset Logging

✔ Tags

✔ Parent Run

✔ Child Runs

✔ Nested Runs

✔ Hyperparameter Tuning

---

# 📸 Artifacts Logged

The project logs:

* Trained Model
* Confusion Matrix
* Source Code
* Training Dataset
* Testing Dataset
* Parameters
* Metrics

---

# 🧠 Hyperparameter Tuning

The project demonstrates GridSearchCV with Random Forest.

Parameters searched:

```python
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 10, 20, 30]
}
```

MLflow logs

* Every trial as a Nested Run
* Best Parameters
* Best Accuracy
* Final Model

---

# 📚 Learning Outcomes

This project helps you understand:

* MLflow Basics
* Experiment Tracking
* Model Management
* Artifact Management
* Dataset Logging
* Hyperparameter Tracking
* Nested Runs
* GridSearchCV Integration
* MLOps Workflow

---

# 📷 MLflow Dashboard

After running

```bash
mlflow ui
```

Visit

```text
http://127.0.0.1:5000
```

You will be able to view:

* Experiments
* Runs
* Parameters
* Metrics
* Artifacts
* Models
* Source Code
* Dataset Information

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

# ⭐ Support

If you found this repository helpful, consider giving it a ⭐ on GitHub.

---

# 👨‍💻 Author

**Tushar**

If you like this project, don't forget to ⭐ the repository.
