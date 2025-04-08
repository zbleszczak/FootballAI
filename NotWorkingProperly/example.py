#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the Football Match Prediction System
"""

import os
import pandas as pd
from thesportsdb_scraper import TheSportsDBScraper
from prediction_model import FootballPredictionModel
from prediction_pipeline import FootballPredictionPipeline

def example_usage():
    """
    Demonstrate how to use the Football Match Prediction System
    """
    print("Football Match Prediction System - Example Usage")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Initialize the pipeline
    pipeline = FootballPredictionPipeline()
    
    # Example 1: Predict matches for English Premier League
    print("\nExample 1: Predicting matches for English Premier League")
    print("-" * 50)
    premier_league_predictions = pipeline.run_prediction_pipeline(
        method='league',
        identifier='English Premier League',
        output_file='output/premier_league_predictions.csv'
    )
    
    # Example 2: Predict matches for a specific team
    print("\nExample 2: Predicting matches for Manchester United")
    print("-" * 50)
    team_predictions = pipeline.run_prediction_pipeline(
        method='team',
        identifier='Manchester United',
        output_file='output/manchester_united_predictions.csv'
    )
    
    # Example 3: Predict matches for the next 14 days
    print("\nExample 3: Predicting matches for the next 14 days")
    print("-" * 50)
    upcoming_predictions = pipeline.run_prediction_pipeline(
        method='date',
        days=14,
        output_file='output/upcoming_predictions.csv'
    )
    
    print("\nAll examples completed. Check the output directory for prediction results.")

if __name__ == "__main__":
    example_usage()
