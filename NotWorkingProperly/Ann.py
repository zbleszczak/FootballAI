#!/usr/bin/env python3
import matplotlib

matplotlib.use("TkAgg")  # Changing backend to avoid matplotlib issues
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
RANDOM_STATE = 42


def load_data():
    test_df = pd.read_csv("premier_test.csv")
    train_df = pd.read_csv("premier_train.csv")
    fixtures_df = pd.read_excel("fixtures_premier.xlsx", sheet_name="Arkusz1")
    return test_df, train_df, fixtures_df


def main():
    # Load data
    test_df, train_df, fixtures_df = load_data()
    print("Data loaded successfully")

    # Print column names in the fixtures file for debugging
    print("Fixtures columns:", fixtures_df.columns.tolist())

    # Define features to use (e.g., betting odds columns) that appear in all datasets
    common_cols = set(test_df.columns) & set(train_df.columns) & set(fixtures_df.columns)
    features = [col for col in common_cols if any(x in col for x in ['B365', 'BW', 'PS', 'WH', 'VC', 'Avg', 'Max'])]

    # Add team names if available
    if 'HomeTeam' in common_cols and 'AwayTeam' in common_cols:
        features = ['HomeTeam', 'AwayTeam'] + features

    print(f"Using {len(features)} common features for prediction")

    # For test and train data, drop rows with missing features and target (FTR)
    test_df = test_df.dropna(subset=features + ['FTR'])
    train_df = train_df.dropna(subset=features + ['FTR'])

    # For fixtures data, do not drop rows; instead perform imputation for numeric columns.
    if len(fixtures_df) == 0:
        print("ERROR: Fixtures dataset is empty. Please check your fixtures_premier.xlsx file.")
        return

    # Keep numeric features (do not include team names)
    numeric_features = [f for f in features if f not in ['HomeTeam', 'AwayTeam']]

    # Show sample of fixtures data (team names) for reference
    print("\nSample from fixtures dataset:")
    print(fixtures_df[['HomeTeam', 'AwayTeam']].head())

    # Fill in missing numeric features in fixtures using median from test_df
    for col in numeric_features:
        if col in fixtures_df.columns:
            median_val = test_df[col].median()
            fixtures_df[col] = fixtures_df[col].fillna(median_val)

    print(f"After preprocessing: Test: {len(test_df)}, Train: {len(train_df)}, Fixtures: {len(fixtures_df)}")

    # Encode target "FTR" (match result) into numeric labels
    le = LabelEncoder()
    le.fit(['H', 'D', 'A'])  # Order: Home win, Draw, Away win
    test_df['FTR_enc'] = le.transform(test_df['FTR'])
    train_df['FTR_enc'] = le.transform(train_df['FTR'])

    # Combine team names for encoding teams
    all_teams = pd.concat([
        test_df['HomeTeam'], test_df['AwayTeam'],
        train_df['HomeTeam'], train_df['AwayTeam'],
        fixtures_df['HomeTeam'], fixtures_df['AwayTeam']
    ]).dropna().unique()
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)

    # Encode team names in each dataset
    for df in [test_df, train_df, fixtures_df]:
        df['HomeTeam_enc'] = team_encoder.transform(df['HomeTeam'])
        df['AwayTeam_enc'] = team_encoder.transform(df['AwayTeam'])

    # Define feature columns: all numeric betting odds and the encoded team names
    feature_cols = [f for f in features if f not in ['HomeTeam', 'AwayTeam']] + ['HomeTeam_enc', 'AwayTeam_enc']

    # Ensure each dataframe has every column in feature_cols
    for col in feature_cols:
        for df in [test_df, train_df, fixtures_df]:
            if col not in df.columns:
                df[col] = 0  # Assign default value if missing

    # Assemble feature matrix and target vector for test and train
    X_test = test_df[feature_cols]
    y_test = test_df['FTR_enc']
    X_train = train_df[feature_cols]
    y_train = train_df['FTR_enc']
    X_fixtures = fixtures_df[feature_cols]

    print(f"Feature dimensions - Test: {X_test.shape}, Train: {X_train.shape}, Fixtures: {X_fixtures.shape}")

    # Perform grid search to find best RandomForest parameters (using test dataset with known outcomes)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    print("Performing grid search...")
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_test, y_test)
    best_model = grid_search.best_estimator_

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Evaluate model on a validation split from test data
    X_test_train, X_val, y_test_train, y_val = train_test_split(
        X_test, y_test, test_size=0.3, random_state=RANDOM_STATE
    )
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))

    # Save validation confusion matrix to file
    val_cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Validation Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('validation_confusion_matrix.png')
    plt.close()

    # Evaluate on train data (from the fixtures period)
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_train_pred, target_names=le.classes_))

    # Save training confusion matrix
    train_cm = confusion_matrix(y_train, y_train_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Training Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('training_confusion_matrix.png')
    plt.close()

    # Make predictions on the fixtures
    fixture_predictions = best_model.predict(X_fixtures)
    fixture_probs = best_model.predict_proba(X_fixtures)
    predicted_results = le.inverse_transform(fixture_predictions)

    # Add prediction results and probabilities to the fixtures dataframe
    fixtures_df['PredictedResult'] = predicted_results
    for i, class_name in enumerate(le.classes_):
        fixtures_df[f'Prob_{class_name}'] = fixture_probs[:, i]

    # Save the fixture predictions to CSV
    fixtures_df.to_csv('fixtures_predictions.csv', index=False)
    print("\nPredictions saved to 'fixtures_predictions.csv'")

    # Additionally, print the predictions to console
    print("\nPredicted fixture results:")
    print(fixtures_df[['HomeTeam', 'AwayTeam', 'PredictedResult']].head())


if __name__ == "__main__":
    main()
