# ==============================================================
# ECON 1611 — Week 10 Class Exercise
# AFL Fantasy: Full Prediction Pipeline with Random Forest
# ==============================================================
# STRUCTURE
#   Part 1 — Clean the Training Data   (mirrors Data_Cleaning_trainingdata.ipynb)
#   Part 2 — Clean the Test Data       (mirrors Data_Cleaning_testdata.ipynb)
#   Part 3 — Build Model and Predict   (mirrors Main_Prediction.ipynb, using RF)
#
# Run all cells top to bottom in order.
# Parts 1 and 2 save cleaned CSVs that Part 3 loads.
# ==============================================================

# %% ── Setup ──────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path, PurePath

from google.colab import drive
drive.mount('/content/drive')

data_path = PurePath('/content/drive/My Drive/Colab Notebooks/AFL/AFL Fantasy Data')
processed_data_path = Path(data_path) / 'Processed'
processed_data_path.mkdir(parents=True, exist_ok=True)

# ==============================================================
# PART 1 — CLEAN TRAINING DATA
# ==============================================================

# %% ── 1.1  Load raw training data ───────────────────────────────────────────
afl_df = pd.read_csv(data_path / 'afl_fantasy_league_training_data.csv')
afl_df

# %% ── 1.2  Inspect: first 20 rows, summary stats, data types ────────────────
with pd.option_context('display.max_columns', None):
    display(afl_df.head(20))
    display(afl_df.describe(include='all'))
afl_df.info()

# %% ── 1.3  Check for duplicate rows ─────────────────────────────────────────
afl_df.duplicated().value_counts()

# %%
afl_df = afl_df[afl_df.duplicated() == False]
print(f'Rows after removing duplicates: {len(afl_df)}')

# %% ── 1.4  Check for missing values ─────────────────────────────────────────
afl_df.isna().sum()

# %%
with pd.option_context('display.max_columns', None):
    display(afl_df[afl_df.isna().any(axis=1)])

# %%
afl_df = afl_df.dropna()
print(f'Rows after dropping NaN: {len(afl_df)}')

# %% ── 1.5  Inspect object (text) columns ────────────────────────────────────
display(afl_df['Date'].value_counts())
display(afl_df['Day'].value_counts())
display(afl_df['Kickoff'].value_counts())
display(afl_df['Home Team'].value_counts())
display(afl_df['Away Team'].value_counts())
display(afl_df['Venue'].value_counts())
display(afl_df['Weather Summary'].value_counts())
display(afl_df['Home team injuries'].value_counts())
display(afl_df['Away team injuries'].value_counts())

# %% ── 1.6  Drop Venue (perfectly correlated with Home Team) ─────────────────
# Every home team always plays at the same venue — Venue adds no new information
afl_df.groupby('Home Team')['Venue'].value_counts()

# %%
del afl_df['Venue']

# %% ── 1.7  Fix team names and standardise weather capitalisation ─────────────
replace_dict_teams = {
    'Pt. Sovereign Pirates': 'Port Sovereign Pirates',
    'Desert Plains Dingos':  'Desert Plains Dingoes'
}
afl_df[['Home Team', 'Away Team']] = afl_df[['Home Team', 'Away Team']].replace(replace_dict_teams)
afl_df['Weather Summary'] = afl_df['Weather Summary'].str.title()

# %% ── 1.8  Binarise injury columns (True/Yes/T/Y → 1, False/No/F/N → 0) ─────
true_values  = ['TRUE', 'YES', 'T', 'Y']
false_values = ['FALSE', 'NO', 'F', 'N']

for col in ['Home team injuries', 'Away team injuries']:
    afl_df.loc[afl_df[col].str.upper().isin(true_values),  col] = 1
    afl_df.loc[afl_df[col].str.upper().isin(false_values), col] = 0

# Recheck
display(afl_df['Home team injuries'].value_counts())
display(afl_df['Away team injuries'].value_counts())

# %% ── 1.9  Convert Date and extract Time from Kickoff ────────────────────────
afl_df['Date'] = pd.to_datetime(afl_df['Date'])
# Dummy date 1/1/2000 keeps the time component while enabling datetime functions
afl_df['Time'] = pd.to_datetime(afl_df['Kickoff']).apply(
    lambda t: t.replace(year=2000, month=1, day=1)
)
del afl_df['Kickoff']

# %% ── 1.10 Outlier boxplots ──────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
for i, col in enumerate(['Pollen count', 'Temperature (°C)', 'Rain (mm)'], 1):
    plt.subplot(1, 3, i)
    afl_df[[col]].boxplot()
plt.tight_layout()
plt.show()

# Also check score totals for negatives
afl_df[['Home Total', 'Away Total']].boxplot()
plt.show()

# %% ── 1.11 Replace outliers with conditional means ──────────────────────────
# Store these thresholds — we apply the SAME values to the test data in Part 2
temp_mean_train   = afl_df[afl_df['Temperature (°C)'] <= 100]['Temperature (°C)'].mean()
pollen_mean_train = afl_df[afl_df['Pollen count'] <= 20]['Pollen count'].mean()
rain_mean_train   = afl_df[afl_df['Rain (mm)'] <= 15]['Rain (mm)'].mean()

afl_df.loc[afl_df['Temperature (°C)'] > 100, 'Temperature (°C)'] = temp_mean_train
afl_df.loc[afl_df['Pollen count'] > 20,      'Pollen count']      = pollen_mean_train
afl_df.loc[afl_df['Rain (mm)'] > 15,         'Rain (mm)']         = rain_mean_train

# %% ── 1.12 Recalculate Home/Away Total from Goals and Points ─────────────────
# The raw data contains some incorrect negative totals — recalculate from goals/points
afl_df['Home Total'] = (afl_df['Home Goals'] * 6) + afl_df['Home Points']
afl_df['Away Total'] = (afl_df['Away Goals'] * 6) + afl_df['Away Points']

# %% ── 1.13 Save cleaned training data to Processed folder ───────────────────
out_train = processed_data_path / 'afl_fantasy_league_training_data-cleaned.csv'
afl_df.to_csv(out_train, index=False)
print(f'Saved cleaned training data: {out_train}')

# ==============================================================
# PART 2 — CLEAN TEST DATA
# ==============================================================
# Same steps as Part 1 with THREE key differences:
#   (1) No score columns in the raw file  →  no score boxplot step
#   (2) No score recalculation step
#   (3) Use training outlier means for replacement (not recalculated from test)

# %% ── 2.1  Load raw test data ────────────────────────────────────────────────
afl_test = pd.read_csv(data_path / 'afl_fantasy_league_test_data.csv')
afl_test

# %% ── 2.2  Inspect ───────────────────────────────────────────────────────────
with pd.option_context('display.max_columns', None):
    display(afl_test.head(20))
    display(afl_test.describe(include='all'))
afl_test.info()

# %% ── 2.3  Remove duplicates and missing values ─────────────────────────────
afl_test = afl_test[afl_test.duplicated() == False]
afl_test = afl_test.dropna()
print(f'Rows after cleaning: {len(afl_test)}')

# %% ── 2.4  Drop Venue, fix team names, standardise weather ──────────────────
del afl_test['Venue']

afl_test[['Home Team', 'Away Team']] = afl_test[['Home Team', 'Away Team']].replace(replace_dict_teams)
afl_test['Weather Summary'] = afl_test['Weather Summary'].str.title()

# %% ── 2.5  Binarise injury columns ──────────────────────────────────────────
for col in ['Home team injuries', 'Away team injuries']:
    afl_test.loc[afl_test[col].str.upper().isin(true_values),  col] = 1
    afl_test.loc[afl_test[col].str.upper().isin(false_values), col] = 0

# %% ── 2.6  Convert Date and extract Time ────────────────────────────────────
afl_test['Date'] = pd.to_datetime(afl_test['Date'])
afl_test['Time'] = pd.to_datetime(afl_test['Kickoff']).apply(
    lambda t: t.replace(year=2000, month=1, day=1)
)
del afl_test['Kickoff']

# %% ── 2.7  Outlier replacement — use training means, not test means ──────────
# Why training means? The model was trained on data with training-based replacements.
# Using different values for test would be inconsistent with what the model learned.
afl_test.loc[afl_test['Temperature (°C)'] > 100, 'Temperature (°C)'] = temp_mean_train
afl_test.loc[afl_test['Pollen count'] > 20,      'Pollen count']      = pollen_mean_train
afl_test.loc[afl_test['Rain (mm)'] > 15,         'Rain (mm)']         = rain_mean_train

# NOTE: No score recalculation — test data has no Home/Away Goals, Points, Total columns

# %% ── 2.8  Save cleaned test data ───────────────────────────────────────────
out_test = processed_data_path / 'afl_fantasy_league_test_data-cleaned.csv'
afl_test.to_csv(out_test, index=False)
print(f'Saved cleaned test data: {out_test}')

# ==============================================================
# PART 3 — BUILD RANDOM FOREST MODEL AND PREDICT
# ==============================================================

# %% ── 3.1  Load cleaned training data ───────────────────────────────────────
afl_df = pd.read_csv(processed_data_path / 'afl_fantasy_league_training_data-cleaned.csv')

with pd.option_context('display.max_columns', None):
    display(afl_df.head(20))
    display(afl_df.describe())
afl_df.info()

# %% ── 3.2  Generate dummies for categorical columns ─────────────────────────
# drop_first=True removes one category per variable to avoid multicollinearity
afl_df = pd.get_dummies(
    afl_df,
    columns=['Home Team', 'Away Team', 'Day', 'Weather Summary'],
    drop_first=True,
    dtype=int
)

# %% ── 3.3  Create x_train (features) and y_train (target) ───────────────────
x_train = afl_df.copy()
x_train.drop(
    columns=['Home Goals', 'Home Points', 'Home Total',
             'Away Goals', 'Away Points', 'Away Total'],
    inplace=True
)

# Date → Julian date number
x_train['Date'] = pd.to_datetime(x_train['Date'], errors='coerce')
x_train['Date'] = x_train['Date'].apply(pd.Timestamp.to_julian_date)

# Time → minutes since midnight
x_train['Time'] = pd.to_datetime(x_train['Time'], errors='coerce')
x_train['Time'] = x_train['Time'].dt.hour * 60 + x_train['Time'].dt.minute

# Target: score difference (positive = home win, negative = away win)
y_train = afl_df['Home Total'] - afl_df['Away Total']

print(f'x_train shape: {x_train.shape}')
print(f'x_train columns:\n{list(x_train.columns)}')
print(f'\ny_train (first 5):\n{y_train.head()}')

# %% ── 3.4  Ten-fold cross-validation with Random Forest ─────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, x_train, y_train, cv=10)

print('CV R² per fold:', cv_scores.round(4))
print(f'Mean CV R²: {cv_scores.mean():.4f}')
print(f'Std CV R²:  {cv_scores.std():.4f}')

plt.figure(figsize=(8, 4))
plt.bar(range(1, 11), cv_scores, color='steelblue', edgecolor='white')
plt.axhline(cv_scores.mean(), color='red', linestyle='--',
            label=f'Mean R² = {cv_scores.mean():.3f}')
plt.xlabel('Fold')
plt.ylabel('R²')
plt.title('10-Fold Cross-Validation — Random Forest (Training Data)')
plt.legend()
plt.tight_layout()
plt.show()

# %% ── 3.5  Fit Random Forest on full training data ──────────────────────────
# cross_val_score does not permanently fit the model.
# We now fit it on ALL training data so it can be used to predict the test set.
rf.fit(x_train, y_train)
print('Random Forest fitted on full training data.')

# %% ── 3.6  In-sample predictions and evaluation ─────────────────────────────
from sklearn.metrics import r2_score, root_mean_squared_error, PredictionErrorDisplay

y_predict_train = rf.predict(x_train)
print(f'Training R²:   {r2_score(y_train, y_predict_train):.4f}')
print(f'Training RMSE: {root_mean_squared_error(y_train, y_predict_train):.2f}')

_ = PredictionErrorDisplay.from_predictions(
    y_true=y_train, y_pred=y_predict_train, kind='actual_vs_predicted'
).plot()
plt.title('Training Data: Actual vs Predicted Score Difference')
plt.show()

# %% ── 3.7  Feature importance — the Random Column reveal ────────────────────
importances = pd.Series(
    rf.feature_importances_, index=x_train.columns
).sort_values(ascending=False)

top15 = importances.head(15)
colors = ['red' if f == 'Random Column' else 'steelblue' for f in top15.index]

plt.figure(figsize=(9, 5))
top15[::-1].plot(kind='barh', color=colors[::-1])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances — Random Forest')
plt.tight_layout()
plt.show()

print(f'Most important feature: {importances.index[0]}  (importance = {importances.iloc[0]:.4f})')

# ==============================================================
# PREPARING AND ALIGNING THE TEST DATA
# ==============================================================

# %% ── 3.8  Load cleaned test data ───────────────────────────────────────────
afl_test_df = pd.read_csv(processed_data_path / 'afl_fantasy_league_test_data-cleaned.csv')
afl_test_df.isnull().sum()

# %% ── 3.9  Generate dummies — drop_first=FALSE for test data ────────────────
# We use drop_first=False to keep ALL categories.
# reindex (below) will remove any that aren't in x_train and add any that are missing.
# Using drop_first=True here risks dropping a DIFFERENT category than training did.
afl_test_df = pd.get_dummies(
    afl_test_df,
    columns=['Home Team', 'Away Team', 'Day', 'Weather Summary'],
    drop_first=False,
    dtype=int
)

x_test = afl_test_df.copy()

# Date → Julian date
x_test['Date'] = pd.to_datetime(x_test['Date'], errors='coerce')
x_test['Date'] = x_test['Date'].apply(pd.Timestamp.to_julian_date)

# Time → minutes since midnight
x_test['Time'] = pd.to_datetime(x_test['Time'], errors='coerce')
x_test['Time'] = x_test['Time'].dt.hour * 60 + x_test['Time'].dt.minute

# %% ── 3.10 Check column mismatch before alignment ───────────────────────────
print('In x_train but missing from x_test:')
display(set(x_train.columns) - set(x_test.columns))
print('In x_test but not in x_train (will be dropped):')
display(set(x_test.columns) - set(x_train.columns))
print(f'\nx_train: {len(x_train.columns)} columns')
print(f'x_test:  {len(x_test.columns)} columns')

# %% ── 3.11 Align x_test columns to exactly match x_train ────────────────────
# reindex adds missing columns as 0 and drops extra columns.
# This is the key step that lets the trained model accept the test data.
x_test = x_test.reindex(labels=x_train.columns, axis='columns', fill_value=0)

print(f'After reindex — x_test: {len(x_test.columns)} columns')
print(f'Columns match x_train: {list(x_train.columns) == list(x_test.columns)}')

# %% ── 3.12 Predict on test data ─────────────────────────────────────────────
# rf was fitted on x_train — now applied to x_test it has never seen
y_predict_test = rf.predict(x_test)
print('Predictions (first 10):', y_predict_test[:10].round(1))

# %% ── 3.13 Load actual results and evaluate ─────────────────────────────────
afl_result_df = pd.read_csv(data_path / 'afl_fantasy_league_test_data_with_result.csv')

rmse_test = root_mean_squared_error(
    y_true=afl_result_df['Point Difference'], y_pred=y_predict_test
)
r2_test = r2_score(
    y_true=afl_result_df['Point Difference'], y_pred=y_predict_test
)

print(f'Test RMSE:           {rmse_test:.2f} points')
print(f'Test R²:             {r2_test:.4f}')
print(f'\nTraining CV mean R²: {cv_scores.mean():.4f}')
print(f'Test R²:             {r2_test:.4f}')
print(f'Overfitting gap:     {cv_scores.mean() - r2_test:.4f}')

# %% ── 3.14 Actual vs predicted plot ─────────────────────────────────────────
_ = PredictionErrorDisplay.from_predictions(
    y_true=afl_result_df['Point Difference'],
    y_pred=y_predict_test,
    kind='actual_vs_predicted'
).plot()
plt.title('Test Data: Actual vs Predicted Score Difference')
plt.show()

# %% ── 3.15 The Random Column scatter — is it actually random? ───────────────
plt.figure(figsize=(7, 5))
plt.scatter(y_predict_train, x_train['Random Column'],
            label='Training data', alpha=0.5)
plt.scatter(afl_result_df['Point Difference'], x_test['Random Column'],
            label='Test data', alpha=0.5)
plt.xlabel('Score Difference')
plt.ylabel('Random Column value')
plt.title('Random Column: correlated in training, random in test')
plt.legend()
plt.show()
# Blue dots (training): visible pattern — Random Column correlates with predictions
# Orange dots (test): random scatter — the correlation was spurious all along

# %% ── 3.16 BONUS: Retrain without Random Column ─────────────────────────────
x_train_nr = x_train.drop(columns=['Random Column'])
x_test_nr  = x_test.drop(columns=['Random Column'])

rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
cv2 = cross_val_score(rf2, x_train_nr, y_train, cv=10)
rf2.fit(x_train_nr, y_train)
y_pred_nr = rf2.predict(x_test_nr)
r2_nr = r2_score(y_true=afl_result_df['Point Difference'], y_pred=y_pred_nr)

print('=== Comparison: with vs without Random Column ===')
print(f'Training CV R²  WITH:    {cv_scores.mean():.4f}')
print(f'Training CV R²  WITHOUT: {cv2.mean():.4f}  ← drops (was spurious)')
print(f'Test R²         WITH:    {r2_test:.4f}')
print(f'Test R²         WITHOUT: {r2_nr:.4f}  ← improves')
print('\nA lower training score + higher test score = better generalisation.')
