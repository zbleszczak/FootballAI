import os
import json
import time
import requests
from datetime import datetime


class FootballDataCache:
    def __init__(self, api_key="331311", league_id=4328, cache_dir="cache"):
        self.api_key = api_key
        self.league_id = league_id
        self.base_url = f"https://www.thesportsdb.com/api/v1/json/{api_key}"
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Cache file paths
        self.matches_cache = os.path.join(cache_dir, f"matches_{league_id}.json")
        self.stats_cache = os.path.join(cache_dir, f"stats_{league_id}.json")

        # Initialize cache data
        self.matches_data = self._load_cache(self.matches_cache)
        self.stats_data = self._load_cache(self.stats_cache)

    def _load_cache(self, file_path):
        """Load data from cache file"""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"Loaded cache from {file_path}")
                return data
            except:
                return {}
        return {}

    def _save_cache(self, file_path, data):
        """Save data to cache file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved cache to {file_path}")

    def download_season_matches(self, season=None):
        """Download all matches for a season"""
        if season is None:
            current_year = datetime.now().year
            season = f"{current_year - 1}-{current_year}"

        url = f"{self.base_url}/eventsseason.php?id={self.league_id}&s={season}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if "events" in data and data["events"]:
                # Store matches with timestamp
                self.matches_data[season] = {
                    "timestamp": time.time(),
                    "matches": data["events"]
                }
                self._save_cache(self.matches_cache, self.matches_data)
                print(f"Downloaded {len(data['events'])} matches for season {season}")
                return data["events"]
        return []

    def download_match_statistics(self, match_id):
        """Download statistics for a specific match"""
        # Check if we already have this match's stats
        if match_id in self.stats_data:
            return self.stats_data[match_id]["stats"]

        url = f"{self.base_url}/lookupeventstats.php?id={match_id}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if "eventstats" in data and data["eventstats"]:
                # Store statistics with timestamp
                self.stats_data[match_id] = {
                    "timestamp": time.time(),
                    "stats": data["eventstats"]
                }
                self._save_cache(self.stats_cache, self.stats_data)
                return data["eventstats"]
        return []

    def download_all_match_statistics(self, season=None):
        """Download statistics for all matches in a season"""
        matches = self.download_season_matches(season)

        total = len(matches)
        for i, match in enumerate(matches):
            match_id = match["idEvent"]
            print(f"Downloading stats for match {i + 1}/{total}: {match['strHomeTeam']} vs {match['strAwayTeam']}")

            # Add delay to avoid rate limiting
            if i > 0 and i % 10 == 0:
                print("Pausing to avoid rate limits...")
                time.sleep(5)

            self.download_match_statistics(match_id)

        print(f"Downloaded statistics for {total} matches")

    def get_match_with_statistics(self, match_id):
        """Get combined match data with statistics"""
        # Find the match in our cached data
        match_data = None
        for season in self.matches_data:
            if season == "timestamp":
                continue
            for match in self.matches_data[season]["matches"]:
                if match["idEvent"] == match_id:
                    match_data = match
                    break

        if not match_data:
            return None

        # Get statistics
        stats = self.stats_data.get(match_id, {}).get("stats", [])

        # Process statistics into a more usable format
        processed_stats = {}
        for stat in stats:
            stat_name = stat.get("strStat")
            if stat_name in ["Total Shots", "Shots on Goal", "Corner Kicks", "Ball Possession"]:
                home_key = f"home_{stat_name.lower().replace(' ', '_')}"
                away_key = f"away_{stat_name.lower().replace(' ', '_')}"
                processed_stats[home_key] = stat.get("intHome", 0)
                processed_stats[away_key] = stat.get("intAway", 0)

        # Combine match data with statistics
        result = {
            "id": match_data["idEvent"],
            "date": match_data["dateEvent"],
            "home_team_id": match_data["idHomeTeam"],
            "away_team_id": match_data["idAwayTeam"],
            "home_team": match_data["strHomeTeam"],
            "away_team": match_data["strAwayTeam"],
            "home_score": match_data.get("intHomeScore"),
            "away_score": match_data.get("intAwayScore"),
        }

        # Add the processed statistics
        result.update(processed_stats)

        return result
