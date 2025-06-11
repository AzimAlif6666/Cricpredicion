Cricket Match Outcome Prediction Using Machine Learning

1. Introduction
This report documents the development of a machine learning model to predict the outcomes of
cricket matches based on team configurations, match metadata, and weather conditions. The
project uses historical cricket data obtained from Cricsheet, enriched with weather data from
Open-Meteo, and leverages a Random Forest classifier to model match results.

2. Dataset Description
The primary dataset for this project is built from publicly available sources:
- Cricsheet Data: Cricsheet provides structured JSON data for thousands of cricket matches. Each
file contains metadata and ball-by-ball information for a single match. For this project, only
match-level metadata was extracted.
- Weather Data: Historical daily weather data was obtained from Open-Meteo using latitude and
longitude coordinates of known cricket venues and the respective match dates.
The final dataset includes the following features for each match:
- match_id, date, venue, city, team1, team2, toss_winner, toss_decision, winner, temp_max,
temp_min, precipitation
Matches lacking a result or critical metadata were excluded. Coordinates were manually mapped for
a set of well-known venues to retrieve weather data. Matches held in unknown venues or with
missing data were skipped or marked accordingly.

3. Exploratory Data Analysis (EDA)
An initial exploratory analysis was conducted to understand the structure and quality of the dataset:- Most matches contained valid entries for participating teams, toss outcomes, and match results.
- Weather data coverage was adequate for matches held in well-known venues.
- Feature value distributions showed significant class imbalance due to certain teams winning more
frequently.
- Categorical features such as team1, team2, and venue required encoding for machine learning.
- Weather variables had weaker but non-negligible correlations with match outcome.

4. Model Selection
The Random Forest algorithm was chosen for this task due to its robustness and ability to handle
mixed data types:
- Handles categorical and continuous variables effectively after encoding
- Provides feature importance metrics for interpretability
- Works well with limited preprocessing
- Resistant to overfitting in medium-sized datasets

5. Feature Engineering
The following preprocessing steps were applied:
- Label Encoding: Categorical columns (team1, team2, toss_winner, toss_decision, venue) were
label-encoded.
- Target Encoding: The match winner column (winner) was also label-encoded.
- Feature Selection: Final features included team1, team2, toss_winner, toss_decision, venue,
temp_max, temp_min, precipitation

6. Model Architecture
The machine learning model used was RandomForestClassifier with 100 estimators and a fixed
random seed. No hyperparameter tuning was applied.

7. Training Process
The model was trained on the cleaned dataset with label encoding applied. The trained model was
saved to disk.

8. Prediction and Output
After training, the model predicted match outcomes on a separate dataset. Output included
predicted winner, probabilities for top 3 teams, and was saved in predictions.csv.

9. Evaluation Metrics
Evaluation used accuracy, classification report, macro/micro F1 scores, and a feature importance
plot. All metrics were computed using scikit-learn.

10. Output Summary
The script produces:
- Trained model: data/match_winner_model.pkl
- Predictions: data/predictions.csv
- Feature importance: data/feature_importance.png
- Logs: logs/model_log.txt
