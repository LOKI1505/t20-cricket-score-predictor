# T20 Cricket Score Predictor

A Streamlit application that predicts the **final first-innings T20 score** using a trained XGBoost regression pipeline.

## Files

- `app.py` — Streamlit interface and prediction logic.
- `requirements.txt` — Python dependencies.
- `t20_score_predictor_improved.joblib` — **add this file from your Colab run**. It is not included in this download.

## Get the model from Colab

Run this after the improved notebook cell that saves the model:

```python
from google.colab import files
files.download('/content/t20_score_predictor_improved.joblib')
```

Put the downloaded `.joblib` in the **same directory** as `app.py`, then upload the project to GitHub.

Do not accidentally upload `t20_score_predictor.joblib`: that is the *original* model and uses a different feature list and saved-file format.

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `.streamlit/config.toml`, and `t20_score_predictor_improved.joblib`.
2. Open https://share.streamlit.io and connect your GitHub account.
3. Choose **Create app**, select your repository and `main` branch, then enter `app.py` as the entrypoint.
4. Choose Python 3.12 in Advanced settings if prompted.
5. Deploy and test a prediction.

### Version note

`joblib` serializes Python objects. A saved model may not load correctly if the deployment's `scikit-learn` or `xgboost` versions differ from those in Colab. For maximum compatibility, execute this in your **training notebook**:

```python
import sklearn, xgboost, joblib, numpy, pandas
print('scikit-learn:', sklearn.__version__)
print('xgboost:', xgboost.__version__)
print('joblib:', joblib.__version__)
print('numpy:', numpy.__version__)
print('pandas:', pandas.__version__)
```

If loading fails, pin `requirements.txt` to the versions reported by your training session and redeploy.

## Model limitations

- Trained on selected men's T20 matches and teams.
- Predicts **first-innings total**, not the winner.
- Works on situations **after 5 overs and before innings end**.
- Expects recent 5-over and 2-over statistics **including extras** within the same rolling windows as training.
- Reports a retrospective held-out match-level evaluation; it is not a live-data feed.
