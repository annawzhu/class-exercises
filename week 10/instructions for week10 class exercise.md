# Instructions for Week 10 Class Exercise
## ECON 1611 — Full Prediction Pipeline with Random Forest

Run these prompts one at a time in a single Colab session, top to bottom in order.
Parts 1 and 2 save cleaned CSV files that Part 3 loads — do not skip ahead.

---

## PART 1 — Clean the Training Data

---

### Exercise 1: Load and Inspect the Training Data

**Paste this prompt into AI:**

> I have a CSV file 'afl_fantasy_league_training_data.csv' in my Google Drive at the path /content/drive/My Drive/Colab Notebooks/AFL/AFL Fantasy Data/. Write Python code to: (1) mount Google Drive and load the file into a DataFrame called afl_df using pandas; (2) print the first 20 rows, describe(include='all'), and info(); (3) check for duplicates with duplicated().value_counts(), then remove them; (4) check for missing values with isna().sum(), display the rows with NaN, then drop them; (5) print the number of rows at each step.

---

### Exercise 2: Fix, Convert, and Save Training Data

**Paste this prompt into AI:**

> Using afl_df (already loaded), write Python code to: (1) delete Venue; (2) replace 'Pt. Sovereign Pirates' with 'Port Sovereign Pirates' and 'Desert Plains Dingos' with 'Desert Plains Dingoes' in Home Team and Away Team; (3) apply .str.title() to Weather Summary; (4) binarise Home team injuries and Away team injuries: TRUE/YES/T/Y to 1, FALSE/NO/F/N to 0; (5) convert Date to datetime, extract Time from Kickoff using a dummy date 1/1/2000, delete Kickoff; (6) boxplot Pollen count, Temperature (°C), Rain (mm) — replace values above 100/20/15 with conditional means and store each mean as temp_mean_train, pollen_mean_train, rain_mean_train; (7) recalculate Home Total = Home Goals*6 + Home Points, same for Away; (8) save to a 'Processed' subfolder as afl_fantasy_league_training_data-cleaned.csv using Path.mkdir(exist_ok=True).

---

## PART 2 — Clean the Test Data

**Before starting:** The test data goes through the same steps as Part 1 with three differences: (1) no score columns exist so there is nothing to recalculate, (2) use the stored means from Part 1 for outlier replacement — do not recalculate from the test data, (3) no score boxplot step.

---

### Exercise 3: Clean and Save Test Data

**Paste this prompt into AI:**

> I have loaded afl_fantasy_league_test_data.csv into afl_test. Apply the same cleaning pipeline as training data: (1) remove duplicates and NaN; (2) delete Venue; (3) replace team name inconsistencies and apply str.title() to Weather Summary; (4) binarise injury columns using the same TRUE/YES/T/Y and FALSE/NO/F/N mapping; (5) convert Date to datetime, extract Time using dummy date 1/1/2000, delete Kickoff; (6) replace outliers using the stored values temp_mean_train, pollen_mean_train, rain_mean_train from my training cleaning step — do NOT recalculate from the test data; (7) do NOT add any score recalculation step; (8) save to the Processed subfolder as afl_fantasy_league_test_data-cleaned.csv.

---

## PART 3 — Build Model and Predict

---

### Exercise 4: Set Up Features and Target

**Paste this prompt into AI:**

> Load 'afl_fantasy_league_training_data-cleaned.csv' from the Processed folder into afl_df. Write Python code to: (1) create dummies for Home Team, Away Team, Day, Weather Summary using pd.get_dummies with drop_first=True and dtype=int; (2) create x_train by copying afl_df and dropping Home Goals, Home Points, Home Total, Away Goals, Away Points, Away Total; (3) convert Date to a Julian date number using pd.Timestamp.to_julian_date; (4) convert Time to minutes since midnight using dt.hour * 60 + dt.minute; (5) create y_train = afl_df['Home Total'] minus afl_df['Away Total'] before the score columns were dropped; (6) print x_train.shape and list(x_train.columns).

---

### Exercise 5: Random Forest — Cross-Validation, Fit, Feature Importance

**Before running — write your prediction: will R² be above 0.5? Above 0.9? What feature do you expect to be most important?**

**Paste this prompt into AI:**

> I have x_train and y_train from AFL Fantasy cleaned training data. Write Python code to: (1) import RandomForestRegressor and cross_val_score; (2) create rf = RandomForestRegressor(n_estimators=100, random_state=42); (3) run 10-fold CV, print each fold score, mean, and std, then plot a bar chart with a dashed red mean line; (4) fit rf.fit(x_train, y_train) on the full training data; (5) extract rf.feature_importances_, sort descending, plot the top 15 as a horizontal bar chart — colour the bar for 'Random Column' red and all others steelblue; (6) print the name and value of the most important feature.

---

### Exercise 6: Prepare Test Data, Predict, and Evaluate

**Note: you are using the same `rf` fitted on training data in Exercise 5 — the model has never seen the test data.**

**Paste this prompt into AI:**

> I have a cleaned test DataFrame loaded from 'afl_fantasy_league_test_data-cleaned.csv' and a fitted RandomForestRegressor rf with x_train. Write Python code to: (1) create dummies using pd.get_dummies with drop_first=False and dtype=int on Home Team, Away Team, Day, Weather Summary; (2) copy to x_test, convert Date to Julian date and Time to minutes since midnight; (3) print columns in x_train not in x_test and vice versa; (4) align with x_test.reindex(labels=x_train.columns, axis='columns', fill_value=0); (5) generate y_predict_test = rf.predict(x_test); (6) load 'afl_fantasy_league_test_data_with_result.csv', calculate r2_score and root_mean_squared_error vs 'Point Difference'; (7) plot actual vs predicted using PredictionErrorDisplay; (8) print training CV mean R² vs test R².

---

## Bonus: Retrain Without Random Column

After Exercise 6, try this follow-up:

> Drop 'Random Column' from both x_train and x_test, create rf2 = RandomForestRegressor(n_estimators=100, random_state=42), rerun 10-fold cross-validation, fit on the trimmed x_train, and predict on the trimmed x_test. Print: (1) new training CV mean R²; (2) new test R²; (3) compare both to the original results with Random Column included and explain what the difference tells us about overfitting.
