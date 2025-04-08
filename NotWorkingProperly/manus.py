import pandas as pd

# Load the dataset
file_path = "combined_football_data.csv"  # Replace with your actual file path
output_file = "cleaned_football_data_with_teams.csv"

# Load the CSV file
df = pd.read_csv(file_path)

# Columns to retain based on your model's features and match context
columns_to_keep = [
    # Team and match context
    'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',

    # Basic odds features
    'B365H', 'B365D', 'B365A',

    # Team form features
    'HomeTeam_FormPts', 'AwayTeam_FormPts',
    'HomeTeam_FormGoalDiff', 'AwayTeam_FormGoalDiff',

    # Derived features
    'FormPtsRatio', 'FormGoalDiffRatio',

    # Betting derived features
    'B365H_Prob_Normalized', 'B365D_Prob_Normalized', 'B365A_Prob_Normalized',
    'HomeWin_DrawRatio', 'HomeWin_AwayWinRatio', 'HomeTeam_IsFavorite',

    # Date features
    'Month', 'DayOfWeek',

    # League info
    'Div'
]

# Ensure only the required columns are kept (ignore missing columns)
df_cleaned = df[[col for col in columns_to_keep if col in df.columns]]

# Save cleaned dataset to a new CSV file
df_cleaned.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}")
