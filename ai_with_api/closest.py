import os
import json
import sys
import time
import requests
from datetime import datetime, timedelta

from ai_with_api.main import FootballPredictionSystem


class FootballDataCache:
    def __init__(self, api_key="331311", league_id=4328, cache_dir="cache"):
        self.api_key = api_key
        self.league_id = league_id
        self.base_url = f"https://www.thesportsdb.com/api/v1/json/{api_key}"
        self.cache_dir = cache_dir
        self.headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Cache file paths
        self.matches_cache = os.path.join(cache_dir, f"matches_{league_id}.json")
        self.stats_cache = os.path.join(cache_dir, f"stats_{league_id}.json")
        self.teams_cache = os.path.join(cache_dir, f"teams_{league_id}.json")
        self.positions_cache = os.path.join(cache_dir, f"positions_{league_id}.json")
        self.upcoming_cache = os.path.join(cache_dir, f"upcoming_{league_id}.json")

        # Initialize cache data
        self.matches_data = self._load_cache(self.matches_cache)
        self.stats_data = self._load_cache(self.stats_cache)
        self.teams_data = self._load_cache(self.teams_cache)
        self.positions_data = self._load_cache(self.positions_cache)
        self.upcoming_data = self._load_cache(self.upcoming_cache)

        # Cache expiration times (in seconds)
        self.cache_expiration = {
            "matches": 86400,  # 24 hours
            "stats": 604800,  # 1 week
            "teams": 604800,  # 1 week
            "positions": 3600,  # 1 hour
            "upcoming": 3600  # 1 hour
        }

    def _load_cache(self, file_path):
        """Load data from cache file"""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"Loaded cache from {file_path}")
                return data
            except Exception as e:
                print(f"Error loading cache from {file_path}: {e}")
                return {}
        return {}

    def _save_cache(self, file_path, data):
        """Save data to cache file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved cache to {file_path}")
        except Exception as e:
            print(f"Error saving cache to {file_path}: {e}")

    def _is_cache_valid(self, cache_type, key=None):
        """Check if cache is still valid based on timestamp"""
        cache_data = getattr(self, f"{cache_type}_data")

        if not cache_data:
            return False

        if key is not None:
            if key not in cache_data:
                return False
            timestamp = cache_data[key].get("timestamp", 0)
        else:
            timestamp = cache_data.get("timestamp", 0)

        current_time = time.time()
        return (current_time - timestamp) < self.cache_expiration[cache_type]

    def get_teams(self):
        """Get teams with caching"""
        if self._is_cache_valid("teams"):
            return self.teams_data.get("teams", [])

        url = f"{self.base_url}/lookup_all_teams.php?id={self.league_id}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            if "teams" in data and data["teams"]:
                self.teams_data = {
                    "timestamp": time.time(),
                    "teams": data["teams"]
                }
                self._save_cache(self.teams_cache, self.teams_data)
                return data["teams"]
        return []

    def get_team_positions(self, season=None):
        """Get team positions with caching"""
        cache_key = season or "current"

        if self._is_cache_valid("positions", cache_key):
            return self.positions_data.get(cache_key, {}).get("positions", {})

        if season is None:
            current_year = datetime.now().year
            season = f"{current_year - 1}-{current_year}"

        url = f"{self.base_url}/lookuptable.php"
        params = {'l': self.league_id, 's': season}

        response = requests.get(url, params=params, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            positions = {}

            if "table" in data and data["table"]:
                for team in data["table"]:
                    team_id = str(team.get("idTeam", ""))
                    if team_id:
                        positions[team_id] = int(team.get("intRank", 0) or 0)

                self.positions_data[cache_key] = {
                    "timestamp": time.time(),
                    "positions": positions
                }
                self._save_cache(self.positions_cache, self.positions_data)
                return positions
        return {}

    def get_season_matches(self, season=None):
        """Get matches for a season with caching"""
        cache_key = season or "current"

        if self._is_cache_valid("matches", cache_key):
            return self.matches_data.get(cache_key, {}).get("matches", [])

        if season is None:
            current_year = datetime.now().year
            season = f"{current_year - 1}-{current_year}"

        url = f"{self.base_url}/eventsseason.php?id={self.league_id}&s={season}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            if "events" in data and data["events"]:
                self.matches_data[cache_key] = {
                    "timestamp": time.time(),
                    "matches": data["events"]
                }
                self._save_cache(self.matches_cache, self.matches_data)
                return data["events"]
        return []

    def get_match_statistics(self, match_id):
        """Get match statistics with caching"""
        if self._is_cache_valid("stats", match_id):
            return self.stats_data.get(match_id, {}).get("stats", [])

        url = f"{self.base_url}/lookupeventstats.php?id={match_id}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            if "eventstats" in data and data["eventstats"]:
                self.stats_data[match_id] = {
                    "timestamp": time.time(),
                    "stats": data["eventstats"]
                }
                self._save_cache(self.stats_cache, self.stats_data)
                return data["eventstats"]
        return []

    def get_upcoming_matches(self):
        """Get upcoming matches directly from API (no caching)"""
        url = f"{self.base_url}/eventsnextleague.php?id={self.league_id}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            data = response.json()
            if "events" in data and data["events"]:
                return data["events"]
        return []

    def check_matches_completeness(self, season=None):
        """Check if all matches for the season are in the cache"""
        if season is None:
            current_year = datetime.now().year
            season = f"{current_year - 1}-{current_year}"

        # Count matches we have in cache for this season
        cache_key = season
        cached_matches = set()

        if cache_key in self.matches_data:
            for match in self.matches_data[cache_key].get("matches", []):
                cached_matches.add(match["idEvent"])

        # Get all matches from API to compare
        url = f"{self.base_url}/eventsseason.php?id={self.league_id}&s={season}"
        response = requests.get(url, headers=self.headers)

        api_matches = set()
        if response.status_code == 200:
            data = response.json()
            if "events" in data and data["events"]:
                for match in data["events"]:
                    api_matches.add(match["idEvent"])

        # Calculate missing matches
        missing_matches = api_matches - cached_matches

        if missing_matches:
            print(f"Found {len(missing_matches)} missing matches in cache for season {season}")
            return False, list(missing_matches)
        else:
            print(f"Cache is complete for season {season}: {len(cached_matches)} matches")
            return True, []

    def update_missing_matches(self, season=None):
        """Update any missing matches in the cache"""
        is_complete, missing_ids = self.check_matches_completeness(season)

        if not is_complete:
            print(f"Updating {len(missing_ids)} missing matches...")

            # Get all matches for the season
            url = f"{self.base_url}/eventsseason.php?id={self.league_id}&s={season}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                data = response.json()
                if "events" in data and data["events"]:
                    # Get current cached matches
                    cache_key = season
                    if cache_key not in self.matches_data:
                        self.matches_data[cache_key] = {
                            "timestamp": time.time(),
                            "matches": []
                        }

                    # Find and add missing matches
                    for match in data["events"]:
                        if match["idEvent"] in missing_ids:
                            self.matches_data[cache_key]["matches"].append(match)

                    # Save updated cache
                    self._save_cache(self.matches_cache, self.matches_data)
                    print(f"Added {len(missing_ids)} missing matches to cache")
                    return True

        return is_complete

    def get_processed_match_statistics(self, match_id):
        """Get processed match statistics in the expected format"""
        raw_stats = self.get_match_statistics(match_id)

        stats = {}
        # Create a mapping between API stat names and our field names
        stat_mapping = {
            "Total Shots": ["home_shots", "away_shots"],
            "Shots on Goal": ["home_shots_on_target", "away_shots_on_target"],
            "Corner Kicks": ["home_corners", "away_corners"],
            "Ball Possession": ["home_possession", "away_possession"],
            "Total passes": ["home_passes_attempted", "away_passes_attempted"],
            "Passes accurate": ["home_passes_completed", "away_passes_completed"]
        }

        for stat in raw_stats:
            stat_name = stat.get("strStat")
            if stat_name in stat_mapping:
                home_key, away_key = stat_mapping[stat_name]

                # Get values, safely handle non-integer values
                try:
                    home_val = int(stat.get("intHome", 0) or 0)
                except (ValueError, TypeError):
                    home_val = 0

                try:
                    away_val = int(stat.get("intAway", 0) or 0)
                except (ValueError, TypeError):
                    away_val = 0

                # Special handling for possession which is a percentage
                if stat_name == "Ball Possession":
                    stats[home_key] = home_val / 100.0
                    stats[away_key] = away_val / 100.0
                else:
                    stats[home_key] = home_val
                    stats[away_key] = away_val

        # Add xG if available (not in the current API response)
        for stat in raw_stats:
            if "xG" in stat.get("strStat", ""):
                stats["home_xg"] = float(stat.get("intHome", 0) or 0)
                stats["away_xg"] = float(stat.get("intAway", 0) or 0)
                break

        return stats


class CachedFootballPredictionSystem(FootballPredictionSystem):
    def __init__(self, api_key="331311", league_id=4328, cache_dir="cache"):
        super().__init__(api_key, league_id)
        self.cache = FootballDataCache(api_key, league_id, cache_dir)

    def get_league_teams(self):
        """Fetch all teams in the specified league using cache"""
        teams = self.cache.get_teams()

        if teams:
            for team in teams:
                team_id = team["idTeam"]
                self.teams_data[team_id] = {
                    "name": team["strTeam"],
                    "form": [],
                    "home_form": [],
                    "away_form": []
                }
            print(f"Successfully loaded {len(self.teams_data)} teams from cache")
            return True
        else:
            print("No teams found in cache or API")
            return False

    def get_past_matches(self, days_back=365):
        """Fetch matches from the past for the specified league using cache"""
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Get current and previous season
        current_year = datetime.now().year
        seasons = [
            f"{current_year - 1}-{current_year}",  # Current season
            f"{current_year - 2}-{current_year - 1}"  # Previous season
        ]

        all_matches = []

        for season in seasons:
            matches = self.cache.get_season_matches(season)

            for match in matches:
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

                        # Get match statistics
                        match_stats = self.fetch_match_statistics(match["idEvent"])
                        match_data.update(match_stats)

                        all_matches.append(match_data)

        # Sort matches by date
        all_matches.sort(key=lambda x: x["date"])
        self.matches_data = all_matches

        print(f"Successfully loaded {len(self.matches_data)} past matches from cache")
        return len(self.matches_data) > 0

    def fetch_match_statistics(self, match_id):
        """Fetch detailed statistics for a specific match using cache"""
        return self.cache.get_processed_match_statistics(match_id)

    def get_team_positions(self, season=None):
        """Fetch team positions using cache"""
        return self.cache.get_team_positions(season)

    def get_upcoming_matches(self):
        """Fetch upcoming matches directly from API without caching"""
        url = f"{self.base_url}/eventsnextleague.php?id={self.league_id}"
        response = requests.get(url, headers=self.headers)

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

                print(f"Successfully loaded {len(upcoming_matches)} upcoming matches from API")
                return upcoming_matches
        else:
            print(f"Failed to fetch upcoming matches: {response.status_code}")

        return []


# Function to initialize or update cache
def update_cache(cache):
    """Update all cache data for faster future runs"""
    print("Updating cache...")

    # Current and previous season
    current_year = datetime.now().year
    seasons = [f"{current_year - 1}-{current_year}", f"{current_year - 2}-{current_year - 1}"]

    # Get teams
    teams = cache.get_teams()
    print(f"Cached {len(teams)} teams")

    # Get team positions
    for season in seasons:
        positions = cache.get_team_positions(season)
        print(f"Cached {len(positions)} team positions for season {season}")

    # Get matches for each season
    for season in seasons:
        matches = cache.get_season_matches(season)
        print(f"Cached {len(matches)} matches for season {season}")

        # Get match statistics with rate limiting
        for i, match in enumerate(matches):
            if match["strStatus"] == "Match Finished":
                match_id = match["idEvent"]
                print(
                    f"Caching statistics for match {i + 1}/{len(matches)}: {match['strHomeTeam']} vs {match['strAwayTeam']}")

                stats = cache.get_match_statistics(match_id)

                # Add delay every 10 requests to avoid rate limiting
                if (i + 1) % 10 == 0:
                    print("Pausing to avoid rate limits...")
                    time.sleep(5)

    # Get upcoming matches
    upcoming = cache.get_upcoming_matches()
    print(f"Cached {len(upcoming)} upcoming matches")

    print("Cache update complete!")


# Example usage in main program
if __name__ == "__main__":
    # Create a cached version of the prediction system
    predictor = CachedFootballPredictionSystem()
    current_year = datetime.now().year
    seasons = [f"{current_year - 1}-{current_year}", f"{current_year - 2}-{current_year - 1}"]
    for season in seasons:
        predictor.cache.update_missing_matches(season)

    if len(sys.argv) > 1 and sys.argv[1] == "--update-cache":
        update_cache(predictor.cache)

    # Continue with normal prediction workflow
    predictor.get_league_teams()
    predictor.get_past_matches()
    predictor.calculate_team_forms(form_length=5)
    predictor.validate_data()
    predictor.train_model()
    predictions = predictor.predict_upcoming_matches()

    # Display team statistics
    team_id = list(predictor.teams_data.keys())[0]  # Pick first available team
    team_name = predictor.teams_data[team_id]["name"]
    team_positions = predictor.get_team_positions()
    stats = predictor.calculate_team_stats(team_id, predictor.matches_data)

    print(f"\nStatistics for team: {team_name}")
    position = team_positions.get(team_id, "Unknown")
    print(f"Position in table: {position}")

    for key, value in stats.items():
        print(f"{key}: {value}")

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
