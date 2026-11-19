from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="T20 Score Predictor",
    page_icon="🏏",
    layout="centered"
)

MODEL_FILE = Path(__file__).parent / "t20_score_predictor_no_city_v2.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


st.title("🏏 T20 Cricket Score Predictor")
st.write("Predict the final first-innings score from the current match situation.")

if not MODEL_FILE.exists():
    st.error("Trained model file not found.")
    st.info(
        "Download `t20_score_predictor_improved.joblib` from your Google Colab "
        "notebook and place it in the same folder as `app.py`."
    )
    st.stop()

try:
    bundle = load_model()
    model = bundle["pipeline"]
    features = bundle["features"]
    target = bundle["target"]

    # Read available team and city names from the trained encoder.
    encoder = model.named_steps["preprocess"].named_transformers_["category"]
    batting_teams, bowling_teams = encoder.categories_
    teams = sorted(set(batting_teams).intersection(set(bowling_teams)))

except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


with st.form("prediction_form"):
    st.subheader("Match details")

    left, right = st.columns(2)

    with left:
        batting_team = st.selectbox("Batting team", teams, index=teams.index("India") if "India" in teams else 0)

    with right:
        bowling_team = st.selectbox("Bowling team", teams, index=teams.index("Australia") if "Australia" in teams else 0)
        
    st.subheader("Current innings")

    left, middle, right = st.columns(3)
    with left:
        current_score = st.number_input("Runs scored", min_value=0, max_value=450, value=85, step=1)
    with middle:
        wickets_out = st.number_input("Wickets lost", min_value=0, max_value=9, value=2, step=1)
    with right:
        overs_completed = st.number_input("Full overs", min_value=5, max_value=19, value=10, step=1)

    balls_in_over = st.slider(
        "Legal balls bowled in the current over",
        min_value=0,
        max_value=5,
        value=0,
        help="For 10.3 overs, choose 10 full overs and 3 balls."
    )

    st.subheader("Recent performance")

    left, right = st.columns(2)
    with left:
        last_5_runs = st.number_input("Runs in last 5 overs", min_value=0, max_value=250, value=44, step=1)
        last_2_runs = st.number_input("Runs in last 2 overs", min_value=0, max_value=150, value=18, step=1)
    with right:
        wickets_last_5 = st.number_input("Wickets lost in last 5 overs", min_value=0, max_value=9, value=1, step=1)
        dots_last_2 = st.number_input("Dot balls in last 2 overs", min_value=0, max_value=12, value=4, step=1)

    submitted = st.form_submit_button("Predict final score", type="primary", use_container_width=True)


if submitted:
    # Basic checks to keep the entered match situation realistic.
    if batting_team == bowling_team:
        st.error("Batting and bowling teams must be different.")
        st.stop()

    if last_2_runs > last_5_runs or last_5_runs > current_score:
        st.error("Recent runs should satisfy: last 2 overs ≤ last 5 overs ≤ current score.")
        st.stop()

    if wickets_last_5 > wickets_out:
        st.error("Wickets in the last 5 overs cannot exceed total wickets lost.")
        st.stop()

    legal_balls = int(overs_completed) * 6 + int(balls_in_over)
    balls_left = 120 - legal_balls
    wickets_left = 10 - int(wickets_out)
    current_run_rate = round(int(current_score) * 6 / legal_balls, 2)

    # These feature names exactly match the improved Colab notebook.
    match_data = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "current_score": int(current_score),
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "current_run_rate": current_run_rate,
        "last_5_overs_runs": int(last_5_runs),
        "last_2_overs_runs": int(last_2_runs),
        "wickets_last_5_overs": int(wickets_last_5),
        "dot_balls_last_2_overs": int(dots_last_2)
    }

    input_df = pd.DataFrame([match_data])[features]

    try:
        prediction = float(model.predict(input_df)[0])

        # The saved bundle tells us whether the model predicts the final
        # score directly or the number of runs still to be scored.
        if target == "remaining":
            prediction += int(current_score)

        prediction = max(prediction, int(current_score))
        prediction = int(np.rint(prediction))

    except Exception as error:
        st.error("Unable to calculate the prediction.")
        st.exception(error)
        st.stop()

    st.divider()
    st.success("Prediction calculated")
    st.metric("Predicted final score", f"{prediction} runs")
    st.metric("Estimated runs remaining", f"{prediction - int(current_score)} runs")

    test_mae = bundle.get("test_mae")
    if test_mae is not None:
        st.caption(
            f"Model test MAE: {float(test_mae):.2f} runs "
            "(evaluated on held-out matches, not a guarantee for this match)."
        )

st.divider()
st.caption(
    "Educational ML project | Men's T20, first innings, after 5 overs. "
    "Predictions are estimates and do not use live match feeds."
)
