import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = '331311'
BASE_URL = f'https://www.thesportsdb.com/api/v1/json/{API_KEY}'
LEAGUE_ID = '4335'  # La Liga ID

def get_upcoming_fixtures():
    url = f'{BASE_URL}/eventsnextleague.php?id={LEAGUE_ID}'
    response = requests.get(url)
    if response.status_code == 200:
        events = response.json().get('events', [])
        df = pd.DataFrame(events)
        df['dateEvent'] = pd.to_datetime(df['dateEvent'])
        df = df.sort_values('dateEvent')
        return df[['dateEvent', 'strHomeTeam', 'strAwayTeam']].head(5)
    else:
        print(f'Error fetching fixtures: {response.status_code}')
        return pd.DataFrame()

def get_team_last_5_games(team_id):
    url = f'{BASE_URL}/eventslast.php?id={team_id}'
    response = requests.get(url)
    if response.status_code == 200:
        events = response.json().get('results', [])
        return events[:5]
    else:
        print(f'Error fetching team stats: {response.status_code}')
        return []

def main():
    # Get upcoming fixtures
    fixtures = get_upcoming_fixtures()
    print("Upcoming Fixtures:")
    print(fixtures)
    print("\n")

    # Get stats for teams in the upcoming fixtures
    teams = set(fixtures['strHomeTeam'].tolist() + fixtures['strAwayTeam'].tolist())
    for team in teams:
        print(f"Last 5 games for {team}:")
        team_id = f'{BASE_URL}/searchteams.php?t={team}'
        team_response = requests.get(team_id)
        if team_response.status_code == 200:
            team_data = team_response.json().get('teams', [])
            if team_data:
                team_id = team_data[0]['idTeam']
                last_5_games = get_team_last_5_games(team_id)
                for game in last_5_games:
                    print(f"{game['dateEvent']}: {game['strHomeTeam']} {game['intHomeScore']} - {game['intAwayScore']} {game['strAwayTeam']}")
        print("\n")

if __name__ == "__main__":
    main()
