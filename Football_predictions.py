import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Danish Superliga Standings",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for better visibility with dark theme
st.markdown("""
<style>
.stApp {
    background-color: #1E1E1E;
    color: #FFFFFF;
}
.stDataFrame {
    background-color: #2D2D2D;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# Display the dashboard header
st.title("⚽ Danish Superliga Standings")
st.markdown("Current league table for the Danish Superliga")

# API Configuration (HARDCODED API KEY)
api_key = "PuzwhFTVi3cBGviHCtjgb8G5080yFLbfIGJ5JBVZBFgzjyrtoS2l5A7rg4DH"  # Your API key
base_url = "https://api.sportmonks.com/v3/football"

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode")


# Function to fetch league seasons
def fetch_league_seasons(league_id):
    seasons_endpoint = f"{base_url}/seasons"
    params = {
        "api_token": api_key,
        "filter[league_id]": league_id
    }

    try:
        response = requests.get(seasons_endpoint, params=params)

        if debug_mode:
            st.sidebar.write(f"Seasons Request URL: {response.url}")
            st.sidebar.write(f"Response Status Code: {response.status_code}")

        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            st.error(f"Failed to fetch seasons. Status code: {response.status_code}")
            if debug_mode:
                st.error(response.text)
            return []
    except Exception as e:
        st.error(f"Error fetching seasons: {str(e)}")
        return []


# Function to fetch standings by season ID
def fetch_standings_by_season(season_id):
    standings_endpoint = f"{base_url}/standings/seasons/{season_id}"
    params = {
        "api_token": api_key,
        "include": "participant"
    }

    try:
        response = requests.get(standings_endpoint, params=params)

        if debug_mode:
            st.sidebar.write(f"Standings Request URL: {response.url}")
            st.sidebar.write(f"Response Status Code: {response.status_code}")

        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            st.error(f"Failed to fetch standings. Status code: {response.status_code}")
            if debug_mode:
                st.error(response.text)
            return []
    except Exception as e:
        st.error(f"Error fetching standings: {str(e)}")
        return []


# Main dashboard function
def create_dashboard():
    # Danish Superliga ID
    league_id = 271

    # First, get the current/latest season for the league
    with st.spinner("Fetching league seasons..."):
        seasons = fetch_league_seasons(league_id)

        if not seasons:
            st.error("No seasons found for Danish Superliga.")
            return

        # Sort seasons by start date to find the current/most recent one
        current_seasons = sorted(seasons, key=lambda x: x.get('starting_at', ''), reverse=True)

        if current_seasons:
            current_season = current_seasons[0]
            season_id = current_season.get('id')
            season_name = current_season.get('name')
            st.success(f"Found season: {season_name}")
        else:
            st.error("Could not determine current season.")
            return

    # Now fetch standings for the current season
    with st.spinner("Fetching standings..."):
        standings_data = fetch_standings_by_season(season_id)

        if not standings_data:
            st.error("No standings data available for the current season.")
            return

        # Process the standings data
        standings_list = []

        # The structure might vary, so we need to handle different formats
        for standing_group in standings_data:
            # Check if this is a standings group (like a league table)
            if 'standings' in standing_group:
                for team in standing_group.get('standings', []):
                    team_data = {
                        'Position': team.get('position', 0),
                        'Team': team.get('participant', {}).get('name', 'Unknown'),
                        'Played': team.get('details', {}).get('games_played', 0),
                        'Won': team.get('details', {}).get('won', 0),
                        'Drawn': team.get('details', {}).get('draw', 0),
                        'Lost': team.get('details', {}).get('lost', 0),
                        'GF': team.get('details', {}).get('goals_scored', 0),
                        'GA': team.get('details', {}).get('goals_against', 0),
                        'GD': team.get('details', {}).get('goal_difference', 0),
                        'Points': team.get('points', 0)
                    }
                    standings_list.append(team_data)

        # Create DataFrame and sort by position
        if standings_list:
            standings_df = pd.DataFrame(standings_list)
            standings_df = standings_df.sort_values('Position')

            # Display the standings table
            st.subheader(f"Current Standings - {season_name}")
            st.dataframe(standings_df)
        else:
            st.warning("Could not process standings data. The format might be different than expected.")

            # Show raw data in debug mode
            if debug_mode:
                st.subheader("Raw Standings Data")
                st.json(standings_data)


# Run the dashboard
if __name__ == "__main__":
    create_dashboard()
