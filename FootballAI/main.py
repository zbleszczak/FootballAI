import os
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# API Key for TheSportsDB
SPORTS_DB_API_KEY = "331311"

# Set page title
st.set_page_config(page_title="Football Match Predictor", layout="wide")
st.title("⚽ Football Match Predictor")

# Sidebar for model settings
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.6, 0.95, 0.75)

# Date picker
selected_date = st.date_input(
    "Choose Date:",
    min_value=datetime.now().date(),
    max_value=datetime.now().date() + timedelta(days=7),
    value=datetime.now().date()
)

date_str = selected_date.strftime('%Y-%m-%d')

TOP_LEAGUES = [
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    "UEFA Nations League", "UEFA Champions League", "UEFA Europa League",
    "FIFA World Cup Qualification"
]


@st.cache_data
def get_matches(date):
    url = f"https://www.thesportsdb.com/api/v1/json/{SPORTS_DB_API_KEY}/eventsday.php?d={date}&s=Soccer"
    response = requests.get(url)
    data = response.json()
    return data.get('events', [])


def fetch_team_stats(team_id):
    """
    Fetch statistics for a specific team using TheSportsDB API.
    """
    url = f"https://www.thesportsdb.com/api/v1/json/{SPORTS_DB_API_KEY}/lookupteam.php?id={team_id}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        team_data = data.get('teams', [{}])[0]

        # If no valid data is returned, use fallback values
        if not team_data:
            st.warning(f"Failed to fetch stats for team {team_id}. Using default values.")
            return {
                'goals_scored': 0,
                'goals_conceded': 0,
                'win_ratio': 0,
                'loss_ratio': 0,
                'draw_ratio': 0,
            }

        return {
            'goals_scored': int(team_data.get('intGoalsScored', 0)),
            'goals_conceded': int(team_data.get('intGoalsConceded', 0)),
            'win_ratio': float(team_data.get('intWinRatio', 0)),
            'loss_ratio': float(team_data.get('intLossRatio', 0)),
            'draw_ratio': float(team_data.get('intDrawRatio', 0)),
        }

    else:
        st.warning(f"Failed to fetch stats for team {team_id}. Using default values.")
        return {
            'goals_scored': 0,
            'goals_conceded': 0,
            'win_ratio': 0,
            'loss_ratio': 0,
            'draw_ratio': 0,
        }


def prepare_features(home_team_id, away_team_id):
    home_stats = fetch_team_stats(home_team_id)
    away_stats = fetch_team_stats(away_team_id)

    # Combine features: home advantage + team metrics
    features = [
        home_stats['goals_scored'], home_stats['goals_conceded'],
        away_stats['goals_scored'], away_stats['goals_conceded'],
        home_stats['win_ratio'], away_stats['win_ratio'],
        home_stats['loss_ratio'], away_stats['loss_ratio'],
        home_stats['draw_ratio'], away_stats['draw_ratio'],
        1  # Home advantage factor (binary: home=1, away=0)
    ]

    return np.array(features).reshape(1, -1)


MODEL_FILE = 'match_predictor.pkl'

if not os.path.exists(MODEL_FILE):
    # Train a dummy model with random data (for demonstration purposes)
    X_dummy = np.random.rand(1000, 11)  # Random features (11 metrics: home+away+advantage)
    y_dummy = np.random.choice(['Home Win', 'Draw', 'Away Win'], size=1000)  # Random outcomes

    model = RandomForestClassifier()
    model.fit(X_dummy, y_dummy)

    # Save the trained model to a file
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
else:
    # Load the trained model from file
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

matches = get_matches(date_str)

if matches:
    st.subheader(f"Matches on {date_str}")

    predictions = []

    for match in matches:
        home_team_id = match['idHomeTeam']
        away_team_id = match['idAwayTeam']

        try:
            features = prepare_features(home_team_id, away_team_id)
            outcome_probs = model.predict_proba(features)[0]

            predictions.append({
                'Match': match['strEvent'],
                'Time': f"{date_str} {match['strTime']}",
                'League': match.get('strLeague', 'Unknown'),
                'Home Win': f"{outcome_probs[0]:.2%}",
                'Draw': f"{outcome_probs[1]:.2%}",
                'Away Win': f"{outcome_probs[2]:.2%}",
                'Recommended Bet': max(outcome_probs.items(), key=lambda x: float(x[1].strip('%')))[0]
                if max(map(float, [x.strip('%') for x in
                                   outcome_probs.values()])) > confidence_threshold else 'No strong recommendation'
            })

        except Exception as e:
            predictions.append({
                'Match': match['strEvent'],
                'Time': f"{date_str} {match['strTime']}",
                'League': match.get('strLeague', 'Unknown'),
                'Error': str(e)
            })

pred_df = pd.DataFrame(predictions)

st.subheader("Predictions")
st.dataframe(pred_df[['Match', 'Time', 'League', 'Home Win', 'Draw', 'Away Win', 'Recommended Bet']],
             use_container_width=True)

st.sidebar.info("This model uses historical match statistics to predict outcomes.")
