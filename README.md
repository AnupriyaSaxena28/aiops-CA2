# AIOps CA-2: Spambase Spam Email Classification

A beginner-friendly ML deployment project: train a spam classifier in Google Colab, serve predictions through FastAPI, package it with Docker, and test it automatically with GitHub Actions.

## Dataset
- File: `spambase_csv.csv`
- 4,601 rows, 57 numerical features, target column `class`
- `0` = legitimate email (ham), `1` = spam
- No missing values in the supplied CSV

## 1. Train in Google Colab (not VS Code)
1. Open `Spambase_Training_Colab.ipynb` in Google Colab.
2. Upload `spambase_csv.csv` to the Colab session using the Files pane.
3. Run all cells from top to bottom.
4. Review accuracy, precision, recall, F1-score, confusion matrix and classification report.
5. Download the generated `model.joblib`.
6. Replace this project's `model.joblib` with the downloaded file.

The saved model is a scikit-learn Pipeline containing `StandardScaler` and `LogisticRegression`.

## 2. Run the API locally in VS Code
Open a terminal in this folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
uvicorn app:app --reload
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

Open:
- `http://127.0.0.1:8000/` — welcome endpoint
- `http://127.0.0.1:8000/health` — health check
- `http://127.0.0.1:8000/docs` — interactive Swagger UI

## 3. Test API
In Swagger UI, expand `POST /predict`, click **Try it out**, paste a JSON body with exactly 57 numeric values, then click **Execute**. The notebook's sample cell uses the dataset's first row as a sanity check.

Example shape (values are placeholders; provide exactly 57 numbers):
```json
{"features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
```

## 4. Run automated tests
With the virtual environment active:
```powershell
pytest -v
```

## 5. Run using Docker Desktop
Make sure Docker Desktop is running. From the project folder:
```powershell
docker build -t spambase-api .
docker run --name spambase-container -p 8000:8000 spambase-api
```
Open `http://127.0.0.1:8000/docs`. To stop the container, press `Ctrl+C` in the terminal or run `docker stop spambase-container` in another terminal. Remove it with `docker rm spambase-container`.

## 6. Push to GitHub
Create a new empty repository named `aiops-spambase-ca2` on GitHub. Do not initialize it with a README if you are pushing this folder's existing README.

From this folder:
```powershell
git init
git add .
git commit -m "Initial AIOps CA-2 spam classification project"
git branch -M main
git remote add origin https://github.com/YOUR-GITHUB-USERNAME/aiops-spambase-ca2.git
git push -u origin main
```
Replace `YOUR-GITHUB-USERNAME` with your GitHub username.

## 7. GitHub Actions CI
The workflow is `.github/workflows/ci.yml`. On every push to `main` and every pull request targeting `main`, it installs Python dependencies, runs `pytest -v`, and builds the Docker image.

On GitHub, open the repository → **Actions** → **Spambase API CI**. A green check means the workflow passed. Open the run to show the test and Docker build logs.

## API response
`POST /predict` returns a prediction (`0` ham or `1` spam), a label, and estimated spam probability. This is a teaching demo using precomputed Spambase features; it does not accept raw email text.
