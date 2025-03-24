import requests
import numpy as np
from datetime import datetime, timedelta

import self
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class FootballPredictionSystem:
    def __init__(self, api_key, league_id):
        self.api_key = api_key  # Your API key is stored here
        self.league_id = league_id
        self.base_url = "https://www.thesportsdb.com/api/v1/json/331311"
        self.headers = {
            "X-API-KEY": self.api_key,  # Your API key is used here in the headers
            "Content-Type": "application/json"
        }
        self.teams_data = {}
        self.matches_data = []
        self.model = None

    def get_league_teams(self):
        """Fetch all teams in the specified league"""
        url = f"{self.base_url}/lookup_all_teams.php?id={self.league_id}"
        response = requests.get(url, headers=self.headers)  # API key used in request

        if response.status_code == 200:
            data = response.json()
            if "teams" in data and data["teams"]:
                for team in data["teams"]:
                    team_id = team["idTeam"]
                    self.teams_data[team_id] = {
                        "name": team["strTeam"],
                        "form": [],
                        "home_form": [],
                        "away_form": []
                    }
                print(f"Successfully loaded {len(self.teams_data)} teams")
                return True
            else:
                print("No teams found for this league")
                return False
        else:
            print(f"Failed to fetch teams: {response.status_code}")
            return False

    def get_past_matches(self, days_back=120):
        """Fetch matches from the past for the specified league"""
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Get current season
        current_year = datetime.now().year
        season = f"{current_year - 1}-{current_year}"

        url = f"{self.base_url}/eventsseason.php?id={self.league_id}&s={season}"
        response = requests.get(url, headers=self.headers)  # API key used in request

        if response.status_code == 200:
            data = response.json()
            if "events" in data and data["events"]:
                for match in data["events"]:
                    # Only include completed matches with scores
                    if match["strStatus"] == "Match Finished" and match["intHomeScore"] and match["intAwayScore"]:
                        match_date = datetime.strptime(match["dateEvent"], "%Y-%m-%d")
                        # Only include matches within our date range
                        if start_date <= match_date <= end_date:
                            match_data = {
                                "id": match["idEvent"],
                                "date": match["dateEvent"],
                                "home_team_id": match["idHomeTeam"],
                                "away_team_id": match["idAwayTeam"],
                                "home_team": match["strHomeTeam"],
                                "away_team": match["strAwayTeam"],
                                "home_score": int(match["intHomeScore"]),
                                "away_score": int(match["intAwayScore"]),
                                "result": self._determine_result(int(match["intHomeScore"]), int(match["intAwayScore"]))
                            }
                            self.matches_data.append(match_data)

                # Sort matches by date
                self.matches_data.sort(key=lambda x: x["date"])
                print(f"Successfully loaded {len(self.matches_data)} past matches")
                return True
            else:
                print("No matches found for this league")
                return False
        else:
            print(f"Failed to fetch matches: {response.status_code}")
            return False

    def _determine_result(self, home_score, away_score):
        """Determine match result (1=home win, 0=draw, 2=away win)"""
        if home_score > away_score:
            return 1  # Home win
        elif home_score == away_score:
            return 0  # Draw
        else:
            return 2  # Away win

    def calculate_team_forms(self, form_length=3):
        """Calculate recent form for all teams"""
        # Create a copy of matches sorted by date
        sorted_matches = sorted(self.matches_data, key=lambda x: x["date"])

        # Initialize form arrays for all teams
        for team_id in self.teams_data:
            self.teams_data[team_id]["form"] = []
            self.teams_data[team_id]["home_form"] = []
            self.teams_data[team_id]["away_form"] = []

        # Process each match to update team forms
        for match in sorted_matches:
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]
            result = match["result"]

            # Skip if team IDs are not in our data
            if home_id not in self.teams_data or away_id not in self.teams_data:
                continue

            # Update home team form
            if result == 1:  # Home win
                self.teams_data[home_id]["form"].append(1)
                self.teams_data[home_id]["home_form"].append(1)
                self.teams_data[away_id]["form"].append(-1)
                self.teams_data[away_id]["away_form"].append(-1)
            elif result == 0:  # Draw
                self.teams_data[home_id]["form"].append(0)
                self.teams_data[home_id]["home_form"].append(0)
                self.teams_data[away_id]["form"].append(0)
                self.teams_data[away_id]["away_form"].append(0)
            else:  # Away win
                self.teams_data[home_id]["form"].append(-1)
                self.teams_data[home_id]["home_form"].append(-1)
                self.teams_data[away_id]["form"].append(1)
                self.teams_data[away_id]["away_form"].append(1)

            # Keep only the most recent matches
            self.teams_data[home_id]["form"] = self.teams_data[home_id]["form"][-form_length:]
            self.teams_data[home_id]["home_form"] = self.teams_data[home_id]["home_form"][-form_length:]
            self.teams_data[away_id]["form"] = self.teams_data[away_id]["form"][-form_length:]
            self.teams_data[away_id]["away_form"] = self.teams_data[away_id]["away_form"][-form_length:]

        teams_with_form = sum(1 for team in self.teams_data.values() if team["form"])
        print(f"Team forms calculated with data for {teams_with_form} teams")

    # Before creating your feature vector, add these definitions
    home_advantage = 1  # Home advantage factor

    # Get team positions from Premier League Standings API
    def get_team_positions(self):
        positions = {}
        url = "https://publicapi.dev/premier-league-standings"  # Example URL
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for team in data["standings"]:
                team_name = team["team"]
                position = team["position"]
                positions[team_name] = position
        return positions

    def calculate_team_stats(self, team_id, matches_data):
        shots = []
        shots_on_target = []
        corners = []
        goals_scored = []
        goals_conceded = []
        possession_values = []
        passes_attempted = []
        passes_completed = []

        # For xG calculation
        shot_distances = []
        shot_angles = []

        for match in matches_data:
            if match["home_team_id"] == team_id:
                # Basic stats
                if "home_shots" in match:
                    shots.append(match["home_shots"])
                if "home_shots_on_target" in match:
                    shots_on_target.append(match["home_shots_on_target"])
                if "home_corners" in match:
                    corners.append(match["home_corners"])
                if "home_possession" in match:
                    possession_values.append(match["home_possession"])
                if "home_passes_attempted" in match:
                    passes_attempted.append(match["home_passes_attempted"])
                if "home_passes_completed" in match:
                    passes_completed.append(match["home_passes_completed"])

                goals_scored.append(match["home_score"])
                goals_conceded.append(match["away_score"])

                # For xG calculation (if available)
                if "home_shot_distances" in match:
                    shot_distances.extend(match["home_shot_distances"])
                if "home_shot_angles" in match:
                    shot_angles.extend(match["home_shot_angles"])

            elif match["away_team_id"] == team_id:
                # Basic stats
                if "away_shots" in match:
                    shots.append(match["away_shots"])
                if "away_shots_on_target" in match:
                    shots_on_target.append(match["away_shots_on_target"])
                if "away_corners" in match:
                    corners.append(match["away_corners"])
                if "away_possession" in match:
                    possession_values.append(match["away_possession"])
                if "away_passes_attempted" in match:
                    passes_attempted.append(match["away_passes_attempted"])
                if "away_passes_completed" in match:
                    passes_completed.append(match["away_passes_completed"])

                goals_scored.append(match["away_score"])
                goals_conceded.append(match["home_score"])

                # For xG calculation (if available)
                if "away_shot_distances" in match:
                    shot_distances.extend(match["away_shot_distances"])
                if "away_shot_angles" in match:
                    shot_angles.extend(match["away_shot_angles"])

        # Calculate shot conversion rate
        shot_conversion_rate = 0
        if shots and sum(shots) > 0:
            shot_conversion_rate = sum(goals_scored) / sum(shots)

        # Calculate shot on target conversion rate
        shot_on_target_conversion_rate = 0
        if shots_on_target and sum(shots_on_target) > 0:
            shot_on_target_conversion_rate = sum(goals_scored) / sum(shots_on_target)

        # Calculate pass completion rate
        pass_completion_rate = 0
        if passes_attempted and sum(passes_attempted) > 0:
            pass_completion_rate = sum(passes_completed) / sum(passes_attempted)

        # Calculate average possession
        avg_possession = sum(possession_values) / len(possession_values) if possession_values else 0

        expected_goals = 0
        if "home_xg" in match and match["home_team_id"] == team_id:
            expected_goals = match["home_xg"]
        elif "away_xg" in match and match["away_team_id"] == team_id:
            expected_goals = match["away_xg"]
        else:
            # Fallback xG calculation if API data isn't available
            expected_goals = sum(goals_scored) * 0.8 if goals_scored else 0

        return {
            "avg_shots": sum(shots) / len(shots) if shots else 0,
            "avg_shots_on_target": sum(shots_on_target) / len(shots_on_target) if shots_on_target else 0,
            "avg_corners": sum(corners) / len(corners) if corners else 0,
            "avg_goals_scored": sum(goals_scored) / len(goals_scored) if goals_scored else 0,
            "avg_goals_conceded": sum(goals_conceded) / len(goals_conceded) if goals_conceded else 0,
            "shot_conversion_rate": shot_conversion_rate,
            "shot_on_target_conversion_rate": shot_on_target_conversion_rate,
            "pass_completion_rate": pass_completion_rate,
            "avg_possession": avg_possession,
            "expected_goals": expected_goals,
        }

    def prepare_features(self):
        """Prepare features for model training"""
        features = []
        targets = []

        # Start from a point where teams have some form history
        start_idx = 5

        # Get team positions once
        team_positions = self.get_team_positions()

        for i in range(start_idx, len(self.matches_data)):
            match = self.matches_data[i]
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]

            # Skip if we don't have data for either team
            if home_id not in self.teams_data or away_id not in self.teams_data:
                continue

            # Skip if teams don't have enough form history
            if not self.teams_data[home_id]["form"] or not self.teams_data[away_id]["form"]:
                continue

            # Calculate features
            home_form_avg = sum(self.teams_data[home_id]["form"]) / len(self.teams_data[home_id]["form"])
            away_form_avg = sum(self.teams_data[away_id]["form"]) / len(self.teams_data[away_id]["form"])

            home_home_form_avg = 0
            if self.teams_data[home_id]["home_form"]:
                home_home_form_avg = sum(self.teams_data[home_id]["home_form"]) / len(
                    self.teams_data[home_id]["home_form"])

            away_away_form_avg = 0
            if self.teams_data[away_id]["away_form"]:
                away_away_form_avg = sum(self.teams_data[away_id]["away_form"]) / len(
                    self.teams_data[away_id]["away_form"])

            # Get team positions
            home_team_name = self.teams_data[home_id]["name"]
            away_team_name = self.teams_data[away_id]["name"]
            home_team_position = team_positions.get(home_team_name, 0)
            away_team_position = team_positions.get(away_team_name, 0)

            # Calculate team stats
            home_team_stats = self.calculate_team_stats(home_id, self.matches_data[:i])
            away_team_stats = self.calculate_team_stats(away_id, self.matches_data[:i])

            home_avg_shots = home_team_stats["avg_shots"]
            away_avg_shots = away_team_stats["avg_shots"]
            home_avg_corners = home_team_stats["avg_corners"]
            away_avg_corners = away_team_stats["avg_corners"]
            home_goals_scored = home_team_stats["avg_goals_scored"]
            away_goals_scored = away_team_stats["avg_goals_scored"]
            home_goals_conceded = home_team_stats["avg_goals_conceded"]
            away_goals_conceded = away_team_stats["avg_goals_conceded"]

            # New features
            home_shot_conversion = home_team_stats["shot_conversion_rate"]
            away_shot_conversion = away_team_stats["shot_conversion_rate"]
            home_possession = home_team_stats["avg_possession"]
            away_possession = away_team_stats["avg_possession"]
            home_xg = home_team_stats["expected_goals"]
            away_xg = away_team_stats["expected_goals"]
            home_pass_completion = home_team_stats["pass_completion_rate"]
            away_pass_completion = away_team_stats["pass_completion_rate"]

            # Define home advantage
            home_advantage = 1

            # Calculate head-to-head record
            h2h_home_wins = 0
            h2h_away_wins = 0
            h2h_draws = 0

            for past_match in self.matches_data[:i]:  # Only consider matches before current one
                if (past_match["home_team_id"] == home_id and past_match["away_team_id"] == away_id) or \
                        (past_match["home_team_id"] == away_id and past_match["away_team_id"] == home_id):
                    if past_match["result"] == 1 and past_match["home_team_id"] == home_id:
                        h2h_home_wins += 1
                    elif past_match["result"] == 2 and past_match["away_team_id"] == away_id:
                        h2h_away_wins += 1
                    elif past_match["result"] == 0:
                        h2h_draws += 1

            # Create feature vector
            feature_vector = [
                home_form_avg,
                away_form_avg,
                home_home_form_avg,
                away_away_form_avg,
                h2h_home_wins,
                h2h_away_wins,
                h2h_draws,
                home_advantage,
                home_team_position,
                away_team_position,
                home_avg_shots,
                away_avg_shots,
                home_avg_shots,  # Using shots as shots_target since we don't have that data
                away_avg_shots,  # Using shots as shots_target since we don't have that data
                home_avg_corners,
                away_avg_corners,
                home_goals_scored,
                away_goals_scored,
                home_goals_conceded,
                away_goals_conceded,
                home_shot_conversion,
                away_shot_conversion,
                home_possession,
                away_possession,
                home_xg,
                away_xg,
                home_pass_completion,
                away_pass_completion
            ]

            features.append(feature_vector)
            targets.append(match["result"])

        return np.array(features), np.array(targets)

    def validate_data(self):
        """Validate data and print diagnostic information"""
        teams_with_form = 0
        form_lengths = []

        for team_id, team_data in self.teams_data.items():
            if "form" in team_data and team_data["form"]:
                teams_with_form += 1
                form_lengths.append(len(team_data["form"]))

        print(f"Teams with form data: {teams_with_form} out of {len(self.teams_data)}")
        if form_lengths:
            print(f"Average form length: {sum(form_lengths) / len(form_lengths):.2f} matches")
            print(f"Min form length: {min(form_lengths)}, Max form length: {max(form_lengths)}")

        # Check match data
        matches_with_results = sum(1 for match in self.matches_data if "result" in match)
        print(f"Matches with results: {matches_with_results} out of {len(self.matches_data)}")

        # Try to prepare features and report size
        X, y = self.prepare_features()
        print(f"Feature matrix shape: {X.shape if len(X) > 0 else 'Empty'}")
        print(f"Target vector shape: {y.shape if len(y) > 0 else 'Empty'}")

    def train_model(self):
        """Train the prediction model"""
        X, y = self.prepare_features()

        if len(X) == 0:
            print("No features available for training")
            self.train_simple_model()  # Fallback to simple model
            return False

        # Add RFECV for feature selection
        selector = RFECV(estimator=RandomForestClassifier(), step=1, cv=5)
        selector.fit(X, y)
        X_selected = selector.transform(X)

        # Print information about selected features
        print(f"Optimal number of features: {selector.n_features_}")
        print(f"Feature ranking: {selector.ranking_}")

        # Define our predictors explicitly
        predictors = [
            'home_form_avg',
            'away_form_avg',
            'home_home_form_avg',
            'away_away_form_avg',
            'h2h_home_wins',
            'h2h_away_wins',
            'h2h_draws',
            'home_advantage',
            'home_team_position',
            'away_team_position',
            'home_avg_shots',
            'away_avg_shots',
            'home_avg_shots_on_target',
            'away_avg_shots_on_target',
            'home_avg_corners',
            'away_avg_corners',
            'home_goals_scored',
            'away_goals_scored',
            'home_goals_conceded',
            'away_goals_conceded',
            'home_shot_conversion',
            'away_shot_conversion',
            'home_possession',
            'away_possession',
            'home_xg',
            'away_xg',
            'home_pass_completion',
            'away_pass_completion'
        ]

        print(f"Training model with predictors: {predictors}")

        # Split data into training and testing sets using the selected features
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")

        # Store the selector for later use in predictions
        self.selector = selector

        return True
    def train_simple_model(self):
        """Train a simpler model with minimal features"""
        # Create simple features: just home/away indicator
        features = []
        targets = []

        for match in self.matches_data:
            if "result" in match:
                # 1 for home advantage
                features.append([1])
                targets.append(match["result"])

        if len(features) == 0:
            print("No features available for training")
            return False

        X = np.array(features)
        y = np.array(targets)

        # Train a simple model
        from sklearn.dummy import DummyClassifier
        self.model = DummyClassifier(strategy="most_frequent")
        self.model.fit(X, y)

        print(f"Simple model trained on {len(features)} matches")
        return True

    def get_upcoming_matches(self):
        """Fetch upcoming matches for prediction"""
        url = f"{self.base_url}/eventsnextleague.php?id={self.league_id}"
        response = requests.get(url, headers=self.headers)  # API key used in request

        upcoming_matches = []

        if response.status_code == 200:
            data = response.json()
            if "events" in data and data["events"]:
                for match in data["events"]:
                    match_data = {
                        "id": match["idEvent"],
                        "date": match["dateEvent"],
                        "home_team_id": match["idHomeTeam"],
                        "away_team_id": match["idAwayTeam"],
                        "home_team": match["strHomeTeam"],
                        "away_team": match["strAwayTeam"]
                    }
                    upcoming_matches.append(match_data)

                print(f"Successfully loaded {len(upcoming_matches)} upcoming matches")
                return upcoming_matches
            else:
                print("No upcoming matches found")
                return []
        else:
            print(f"Failed to fetch upcoming matches: {response.status_code}")
            return []

    def predict_match(self, home_team_id, away_team_id):
        """Predict the outcome of a specific match"""
        if not self.model:
            print("Model not trained yet")
            return None

        if home_team_id not in self.teams_data or away_team_id not in self.teams_data:
            print("Team data not available")
            return None

        # Calculate features for this match
        home_form_avg = sum(self.teams_data[home_team_id]["form"]) / len(self.teams_data[home_team_id]["form"]) if \
        self.teams_data[home_team_id]["form"] else 0
        away_form_avg = sum(self.teams_data[away_team_id]["form"]) / len(self.teams_data[away_team_id]["form"]) if \
        self.teams_data[away_team_id]["form"] else 0

        home_home_form_avg = 0
        if self.teams_data[home_team_id]["home_form"]:
            home_home_form_avg = sum(self.teams_data[home_team_id]["home_form"]) / len(
                self.teams_data[home_team_id]["home_form"])

        away_away_form_avg = 0
        if self.teams_data[away_team_id]["away_form"]:
            away_away_form_avg = sum(self.teams_data[away_team_id]["away_form"]) / len(
                self.teams_data[away_team_id]["away_form"])

        # Calculate head-to-head record
        h2h_home_wins = 0
        h2h_away_wins = 0
        h2h_draws = 0

        for past_match in self.matches_data:
            if (past_match["home_team_id"] == home_team_id and past_match["away_team_id"] == away_team_id) or \
                    (past_match["home_team_id"] == away_team_id and past_match["away_team_id"] == home_team_id):
                if past_match["result"] == 1 and past_match["home_team_id"] == home_team_id:
                    h2h_home_wins += 1
                elif past_match["result"] == 2 and past_match["away_team_id"] == away_team_id:
                    h2h_away_wins += 1
                elif past_match["result"] == 0:
                    h2h_draws += 1

        # Get team positions
        team_positions = self.get_team_positions()
        home_team_name = self.teams_data[home_team_id]["name"]
        away_team_name = self.teams_data[away_team_id]["name"]
        home_team_position = team_positions.get(home_team_name, 0)
        away_team_position = team_positions.get(away_team_name, 0)

        # Calculate team stats
        home_team_stats = self.calculate_team_stats(home_team_id, self.matches_data)
        away_team_stats = self.calculate_team_stats(away_team_id, self.matches_data)

        home_avg_shots = home_team_stats["avg_shots"]
        away_avg_shots = away_team_stats["avg_shots"]
        home_avg_corners = home_team_stats["avg_corners"]
        away_avg_corners = away_team_stats["avg_corners"]
        home_goals_scored = home_team_stats["avg_goals_scored"]
        away_goals_scored = away_team_stats["avg_goals_scored"]
        home_goals_conceded = home_team_stats["avg_goals_conceded"]
        away_goals_conceded = away_team_stats["avg_goals_conceded"]

        # New features
        home_shot_conversion = home_team_stats["shot_conversion_rate"]
        away_shot_conversion = away_team_stats["shot_conversion_rate"]
        home_possession = home_team_stats["avg_possession"]
        away_possession = away_team_stats["avg_possession"]
        home_xg = home_team_stats["expected_goals"]
        away_xg = away_team_stats["expected_goals"]
        home_pass_completion = home_team_stats["pass_completion_rate"]
        away_pass_completion = away_team_stats["pass_completion_rate"]

        # Create feature vector with all features
        feature_vector = [
            home_form_avg,
            away_form_avg,
            home_home_form_avg,
            away_away_form_avg,
            h2h_home_wins,
            h2h_away_wins,
            h2h_draws,
            1,  # Home advantage factor
            home_team_position,
            away_team_position,
            home_avg_shots,
            away_avg_shots,
            home_avg_shots,  # Using shots as shots_target since we don't have that data
            away_avg_shots,  # Using shots as shots_target since we don't have that data
            home_avg_corners,
            away_avg_corners,
            home_goals_scored,
            away_goals_scored,
            home_goals_conceded,
            away_goals_conceded,
            home_shot_conversion,
            away_shot_conversion,
            home_possession,
            away_possession,
            home_xg,
            away_xg,
            home_pass_completion,
            away_pass_completion
        ]

        feature_vector = [feature_vector]  # Convert to 2D array
        if hasattr(self, 'selector'):
            feature_vector = self.selector.transform(feature_vector)

        # Make prediction
        try:
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]

            result_mapping = {
                1: "Home Win",
                0: "Draw",
                2: "Away Win"
            }

            return {
                "prediction": result_mapping[prediction],
                "probabilities": {
                    "Home Win": probabilities[list(self.model.classes_).index(1)] if 1 in self.model.classes_ else 0,
                    "Draw": probabilities[list(self.model.classes_).index(0)] if 0 in self.model.classes_ else 0,
                    "Away Win": probabilities[list(self.model.classes_).index(2)] if 2 in self.model.classes_ else 0
                }
            }
        except:
            # If using simple model or classes are different
            prediction = self.model.predict([feature_vector])[0]

            result_mapping = {
                1: "Home Win",
                0: "Draw",
                2: "Away Win"
            }
            return {
                "prediction": result_mapping[prediction],
                "probabilities": {
                    "Home Win": 0.33,
                    "Draw": 0.33,
                    "Away Win": 0.33
                }
            }

    def predict_upcoming_matches(self):
        """Predict all upcoming matches"""
        upcoming_matches = self.get_upcoming_matches()
        predictions = []

        for match in upcoming_matches:
            home_id = match["home_team_id"]
            away_id = match["away_team_id"]

            prediction = self.predict_match(home_id, away_id)
            if prediction:
                match_prediction = {
                    "match_id": match["id"],
                    "date": match["date"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "prediction": prediction["prediction"],
                    "home_win_probability": prediction["probabilities"]["Home Win"],
                    "draw_probability": prediction["probabilities"]["Draw"],
                    "away_win_probability": prediction["probabilities"]["Away Win"]
                }
                predictions.append(match_prediction)

        # Sort by confidence (highest probability)
        if predictions:
            predictions.sort(
                key=lambda x: max(x["home_win_probability"], x["draw_probability"], x["away_win_probability"]),
                reverse=True)

        return predictions


# Example usage - PUT YOUR API KEY HERE
if __name__ == "__main__":
    # Initialize the prediction system with your API key and league ID
    # Premier League ID: 4328
    predictor = FootballPredictionSystem(api_key="331311",
                                         league_id=4332)  # <-- REPLACE "YOUR_API_KEY_HERE" with your actual API key

    # Load data and train model
    predictor.get_league_teams()
    predictor.get_past_matches(days_back=365)  # Get matches from the past year
    predictor.calculate_team_forms()

    # Validate data to check if we have enough for training
    predictor.validate_data()

    # Train the model
    predictor.train_model()

    # Make predictions for upcoming matches
    predictions = predictor.predict_upcoming_matches()

    # Display predictions
    print("\nUpcoming Match Predictions (Sorted by Confidence):")
    print("=" * 80)
    for pred in predictions:
        print(f"Date: {pred['date']}")
        print(f"Match: {pred['home_team']} vs {pred['away_team']}")
        print(f"Prediction: {pred['prediction']}")
        print(
            f"Probabilities: Home Win: {pred['home_win_probability']:.2f}, Draw: {pred['draw_probability']:.2f}, Away Win: {pred['away_win_probability']:.2f}")
        print("-" * 80)
