# SPY volatility regime classifier
# Notes are for me im lowk learning as I go along
#Goal:
#   Predict if the next Month(21 trading day) of SPY
#   Will be a High or Low volatility month

# Steps
#   1  - Import libraries
#   2  - Settings
#   3  - Download SPY data
#   4  - Calculate daily returns
#   5  - Build features (the inputs the model learns from)
#   6  - Create the label (the answer the model must predict)
#   7  - Clean the data and split into X (features) and y (label)
#   8  - Build the model (scaler + neural network)
#   9  - Time-series cross-validation (test on the future only)
#   10 - Final test on the most recent 20% of data
#   11 - Plot the results
#   12 - Predict the regime for the next month

#1 Import Libraries

import os   #creates the output folder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf

from sklearn.preprocessing import StandardScaler #puts features on the same scale
from sklearn.neural_network import MLPClassifier # the neural network (Multi-Layer Perceptron)
from sklearn.pipeline import Pipeline       # chains scaler + model tg
from sklearn.model_selection import TimeSeriesSplit #Cross validation for time series
from sklearn.metrics import (
    accuracy_score,         # % of predictions that are correct
    balanced_accuracy_score,  # accuracy that is fair when one class is bigger
    roc_auc_score,              # how well the model ranks high vs low vol days (.5 = random)
    confusion_matrix,          # table of right vs wrong predictions
    classification_report        # precision/ recall for each class
)

print("Step 1 done: Libraries imported")


#2 Settings

TICKER = "SPY"
START_DATE = "2000-01-01"
HORIZON = 21            #predicts abt next 21 trading days (abt 1 month)
THRESHOLD_WINDOW = 252  # 252 days td(1 year) used to determins "normal" vol
N_SPLITS = 5                # number of cross-validation folds
TEST_FRACTION = 0.20        # keep the last 20% of data for the final test
RANDOM_STATE = 42           # fixed seed so you get the same results every run
OUTPUT_DIR = "outputs"      # folder where charts and results are saved

os.makedirs(OUTPUT_DIR, exist_ok=True)      #creates the folder and no error if it alr exits

print("Step 2 done: settings ready")

#3 Download SPY data

# yf.download gets one row per trading day with Open, High, Low, Close, Volume.
# auto_adjust=True adjusts prices for dividends and splits,
# so a dividend payment does not look like a price drop.

df = yf.download(TICKER, start=START_DATE, auto_adjust=True, progress=False)

# Downloaded after to try boosting model scores
vix = yf.download("^VIX", start = START_DATE, auto_adjust = True, progress = False)["Close"]
if isinstance(vix, pd.DataFrame):
    vix = vix.iloc[:, 0]            # just a single column
vix = vix.reindex(df.index).ffill()     #lines VIX dates with SPY's and fill any gaps

# yfinance give two-level column names like ("Close", "SPY").
# This line keeps only the first level, so we can simply write df["Close"].

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Low", "Close", "Volume"]]  #only columns we need
df = df.dropna()    #obv

print(f"Step 3 done: downloaded {len(df):,} days "
      f"({df.index[0].date()} to {df.index[-1].date()})")
print(df.tail())                                                #Shows last 5 rows to check if data is good


#4 Calculates daily returns

# Log return = log(today's price / yesterday's price).
# think log(x/y)=log(x)-log(y)
# we use logs since they add up nice 1.10 * .90 =.99 
#thats a 1% loss and that stacks also its apparently standard input for vol calculations

log_ret = np.log(df["Close"]).diff()    # .diff() = today minus yesterday
 
# The first day has no yesterday in the matrix so its return will be empty
print("Step 4 done: daily log returns calculated")
print(log_ret.tail())

#5 Build features

# features = info the model looks at to make its prediction 

#VERY IMPORTANT RULE (no data leakage):
#   Every feature on day t must use ONLY data from day t or earlier.
#   .rolling(w) looks BACKWARD over the last w days, so it is safe.

features = pd.DataFrame(index=df.index)  # empty table, same date index as historical price data so everything stays aligned

# 5a Todays returns
features["ret_1d"] = log_ret   # stores 1 day log return as the foundational feature

# 5b Rolling volatility
# Volatility = standard deviation(sigma) of daily returns.  think stats
# Multiply by sqrt(252) to "annualize" it (convert daily to yearly vol).
for w in [5, 10, 21, 63]:               # 1w 2w 1m 1quarter
    features[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

# 5c Volatility ratios
# is vol rising or falling
# ratio > 1 means short term is higher than long term vol (vol is rising)
features["vol_ratio_5_21"] = features["vol_5d"] / features["vol_21d"]
features["vol_ratio_21_63"] = features["vol_21d"] / features["vol_63d"]

# 5d Parkinson Volatility (Intraday range)
# close to close returns miss big moves INSIDE the day, ts uses high-low range to capture that
## Formula: sqrt( average of (log(High/Low))^2  /  (4 * log(2)) )
high_low_sq = np.log(df["High"] / df["Low"]) ** 2
features["parkinson_vol_21d"] = (
    np.sqrt(high_low_sq.rolling(21).mean() / (4 * np.log(2))) * np.sqrt(252)
)
# 5e downside volatility
# Vol of losing days only. markets usually get vol when they fall(investor fear)
# .where(log return< 0, 0) keeps the neg return and sets positive ones to 0
negative_returns = log_ret.where(log_ret < 0,0.0)
features["downside_vol_21d"] = (
    np.sqrt((negative_returns ** 2).rolling(21).mean()) * np.sqrt(252)
)

# 5f Distribution shape
# mean = avg. return ( is market going up or down)
# skew = is the distribution lopsided? neg skew = more big down days
# kurtosis = fat tails mean more often extreme days
for w in [21, 63]:
    features[f"mean_ret_{w}d"] = log_ret.rolling(w).mean()
    features[f"skew_{w}d"] = log_ret.rolling(w).skew()
    features[f"kurt_{w}d"] = log_ret.rolling(w).kurt()

# 5g tail risk (Value at risk(VaR))
# 5th percentile return of the last 63 days =  bad days level
features["q05_ret_63d"] = log_ret.rolling(63).quantile(0.05)

# 5h drawdown and trend
# drawdown_63d: how far the price is below its highest price of the last 63 days
#  0 at high and -0.1 = 10% below high
# dist_ma_50: how far the price is above/below its 50-day moving average
features["drawdown_63d"] = df["Close"] / df["Close"].rolling(63).max() - 1  
features["dist_ma_50"] = df["Close"] / df["Close"].rolling(50).mean() - 1
  # .rolling(63).max() finds highest price peak over last 3 months. 
# Dividing the current close price by that peak and subtracting 1 calculates percentage distance below the peak.

# 5i Volume Shock
# Z score: how unusual is today's volume compared to the last 21 days?
# 0 = normal, +2 = much higher than normal (a sign of stress)
vol_mean = df["Volume"].rolling(21).mean()
vol_std = df["Volume"].rolling(21).std()
features["volume_z_21d"] = (df["Volume"] - vol_mean) / vol_std

 #5j (added after) VIX: markets expected volatility
features["vix_level"] = vix                 #today's VIX
features["vix_change_5d"] = vix.diff(5)     # how much VIX moved over the last week
features["vix_vs_realized"] = (vix/ 100) /features["vol_21d"]   # expected vol vs recent actual

FEATURE_COLS = list(features.columns)       # save the list of feature names
print(f"Step 5 done: built {len(FEATURE_COLS)} features")
print(FEATURE_COLS)


#6 Creating the label (High vol = 1 Low vol = 0)

# Label is the "answer" the model will try to predict
# Question I ask on each day T:
#   "Will the vol of the NEXT 21 days ne higher than normal"
#
#   future_vol = volatility from day t+1 to day t+21     (the future)
#   threshold  = median volatility of the PAST 252 days  (what is "normal")
#
#   label = 1 (HIGH vol) if future_vol > threshold
#   label = 0 (LOW vol)  otherwise
# Why the PAST year median and not the median of ALL data? 
# The median of all data would include future years. On day t, nobody knows that number yet -> that would be data leakage.

vol_21 = log_ret.rolling(HORIZON).std() * np.sqrt(252)  # 21-day vol on each day

# shift(-21) moves value 21 rows UP so on day t we see vol of days t+1,..., t+21
# this is the ONLY place where we look into the future and its the answer(label)
future_vol = vol_21.shift(-HORIZON)

# median of the last 252 days of 21 day vol (only past data)
threshold = vol_21.rolling(THRESHOLD_WINDOW).median()
regime = (future_vol > threshold).astype(float)     # True/false -> 1/0

# last 21 days dont have a fututre yet and the first year doesnt have data(bcs were living it rn)
# set these to NaN so our neural network doesnt train on incomplete data
regime[future_vol.isna() | threshold.isna()] = np.nan

print("Step 6 done: labels created")

#7 clean the data amd split into x and y

# put featurs and labels in one table so the rows line up by date
data = features.copy()
data["future_vol"] = future_vol
data["threshold"] = threshold
data["regime"] = regime

# if sum divids to infinity treat them as missing
data = data.replace([np.inf, -np.inf], np.nan)

#saving the most recent day that has all features but its labels r unkown (future hasnt happened yet) so we use it at the end to predict next month
latest_row = data[FEATURE_COLS].dropna().iloc[[-1]]   #need a double bracket here or I get 1d array error sum to do with pandas

data = data.dropna() #deleting rows with missing stuff

X = data[FEATURE_COLS]                  # X = inputs (features)
y = data["regime"].astype(int)          # y = answer (0 or 1)
 
print(f"Step 7 done: {len(X):,} usable days")
print(f"  High-vol days: {y.mean():.1%}")
print(f"  Low-vol days:  {1 - y.mean():.1%}")

#8 Building the model

#model has 2 parts joined in the pipeline

#   1. StandardScaler -> rescales every feature to mean 0, std 1.
#      Neural networks learn badly when one feature is 0.01 and another is 50,000,000.
#
#   2. MLPClassifier -> the neural network.
#
#  why the pipeline?
#   The scaler must learn its mean/std from TRAINING data only.
#   If we scaled the whole dataset first, the scaler would "see" the test data
#   data leakage. The Pipeline fits the scaler on the training part automatically.

def build_model():
    """Return a fresh, untrained model. Need a new one for every fold"""
    neural_net = MLPClassifier(
        hidden_layer_sizes=(32,16),  # 2 hidden layers: 32 neuronss then 16 neurons
        activation="relu",            # standard activation function
        alpha = 0.001,                  # L2 regularization: stops the model memorizing noise
        learning_rate_init = 0.001,     #how big each learning step is, how aggresively model updates its weights
        max_iter = 500,              # caps training to 500 complete epochs through the data set
        early_stopping = True,          #stops when the model isnt improving and is checked on last 10% of data
        validation_fraction = 0.1,
        n_iter_no_change = 20,              # stop after 20 epochs with no impovement
        random_state = RANDOM_STATE,            #seeds the random numb generator used to initialize the neural networks starting weight
        # running the exact same code twice will yield slightly different models and backtest results. Seeding ensures full scientific reproducibility
        )
    return Pipeline([
            ("scaler", StandardScaler()),   # part 1
            ("mlp", neural_net),            # part 2
        ])

print("Step 8 done: model defined")
print(build_model())


#9 Time series cross-validation
# cant use standard cross-validation for stocks since their time dependent you could accidently train the model with data from 2020 while being tested on 2015
# meaning the model is using the future to predict the past which is cheating(data leakage)
# time series respects flow of time and strictly tests past data  and tests on fututre
#  
#   Fold 1: [TRAIN]                gap [TEST]
#   Fold 2: [TRAIN.......]         gap [TEST]
#   Fold 3: [TRAIN..............]  gap [TEST]
#
# Why gap=21?
#   The label on the LAST training day looks 21 days into the future,
#   which is inside the test period. Skipping 21 days removes that overlap.

# We also compare against a naive baseline
#   "Next month will be like this month."
#   (High vol now -> predict high vol next month.)
#   A model is only useful if it beats a naive baseline because a simple, zero-cost rule already captures the obvious structural properties of the data, 
# meaning any new model must prove it adds genuine predictive value worth its complexity.

tscv = TimeSeriesSplit(n_splits = N_SPLITS, gap = HORIZON)
cv_rows = []

print(f"Step 9: cross-validation ({N_SPLITS} folds, gap = {HORIZON} days)")
# below  telling the computer: "Take my historical timeline and figure out the exact indexes(Train/Test) to cut it so I can run my 5 sequential practice rounds."
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start = 1):     #loops through splits generated by tscv keeping count oof which fold were on
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    model = build_model()       #func from step 8 to spawn brand new untrained neural network and scalar pipline so it doesnt remember past folds
    model.fit(X_train, y_train)         #model learns weights using only X/y_train

    proba = model.predict_proba(X_test)[:, 1]       # instead of 0/1 the nn outputs probability(lowk dont understand how)
    pred = (proba >= 0.5).astype(int)               #converts prob into hard binary if model > 50% it will be 1 if not 0

    #Naive baseline- is current 21-day volatility above the threshold?
    naive_pred = (X_test["vol_21d"] > data["threshold"].iloc[test_idx]).astype(int)

    #this dictionary compiles exact date boundaries of the fold, alongside core quantative metrics
    result = {
                "fold": fold,
                "train_end": X.index[train_idx[-1]].date(),
                "test_start": X.index[test_idx[0]].date(),
                "test_end": X.index[test_idx[-1]].date(),
                "accuracy": accuracy_score(y_test, pred),
                "balanced_acc": balanced_accuracy_score(y_test, pred),              #Balanced accuracy - ml evaluation metric used to measure classification performance
                "roc_auc": roc_auc_score(y_test, proba),                            #area under ROC curve measuring how well model ranks probabilities
                "naive_balanced_acc": balanced_accuracy_score(y_test, naive_pred),      #score for simple baseline(look above)
            }
    cv_rows.append(result)      #saves this folds results into the container list

    print(f"  Fold {fold}: test {result['test_start']} -> {result['test_end']} | "
            f"model {result['balanced_acc']:.3f} vs naive {result['naive_balanced_acc']:.3f} | "
          f"AUC {result['roc_auc']:.3f}")

cv_results = pd.DataFrame(cv_rows)
cv_results.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)

print(f"  AVERAGE balanced accuracy: model {cv_results['balanced_acc'].mean():.3f} "
      f"vs naive {cv_results['naive_balanced_acc'].mean():.3f}")
print("Step 9 done")

#10 Final test on most recent data

# last 20% of data is the "final exam"
#Train once on everything before it (except the 21 day gap), test once on it.
#Most realistic estimate of how model will do in real life

split_point = int(len(X) * (1 - TEST_FRACTION))     # row where the test period starts

final_train_idx = np.arange(0, split_point - HORIZON)       #stop 21 days early
final_test_idx = np.arange(split_point, len(X))

final_model = build_model()
final_model.fit(X.iloc[final_train_idx], y.iloc[final_train_idx])

test_proba = final_model.predict_proba(X.iloc[final_test_idx])[:, 1]
test_pred = (test_proba >= 0.5).astype(int)
y_final_test = y.iloc[final_test_idx]
 
print(f"Step 10: final test from {X.index[final_test_idx[0]].date()} "
      f"to {X.index[final_test_idx[-1]].date()} ({len(final_test_idx):,} days)")
print(f"  Accuracy:          {accuracy_score(y_final_test, test_pred):.3f}")
print(f"  Balanced accuracy: {balanced_accuracy_score(y_final_test, test_pred):.3f}")
print(f"  ROC AUC:           {roc_auc_score(y_final_test, test_proba):.3f}")

# Naive baseline on the SAME final-test days "next month = this month vol"
naive_test_pred = (X["vol_21d"].iloc[final_test_idx] > data["threshold"].iloc[final_test_idx]).astype(int)
naive_test_bal = balanced_accuracy_score(y_final_test, naive_test_pred)
model_test_bal = balanced_accuracy_score(y_final_test, test_pred)
print(f" Naive baseline balanced accuracy: {naive_test_bal:.3f} (model: {model_test_bal:.3f})")

#                 predicted LOW   predicted HIGH
#   actual LOW        [ a              b ]
#   actual HIGH       [ c              d ]
# a and d = correct, b and c = mistakes
print("  Confusion matrix (rows = actual, columns = predicted) [LOW, HIGH]:")
print(confusion_matrix(y_final_test, test_pred))
print(classification_report(y_final_test, test_pred, target_names=["Low vol", "High vol"]))

#11 Plot the results

test_dates = X.index[final_test_idx]

# Chart 1: three panels for the final test period
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: SPY price, with actual high-vol periods shaded red
price = df["Close"].loc[test_dates]
is_high = (y_final_test == 1).values
axes[0].plot(test_dates, price, color="black", linewidth=1)
axes[0].fill_between(test_dates, price.min(), price.max(), where=is_high,
                     color="red", alpha=0.15, label="Actual HIGH vol")
axes[0].set_title(f"{TICKER} price - final test period")
axes[0].legend(loc="upper left")

# Panel 2: future volatility vs the threshold
axes[1].plot(test_dates, data["future_vol"].loc[test_dates], label="Next 21-day volatility")
axes[1].plot(test_dates, data["threshold"].loc[test_dates], "--", label="Threshold (past-year median)")
axes[1].set_title("Volatility vs threshold (above the dashed line = HIGH vol)")
axes[1].legend(loc="upper left")

# Panel 3: model's probability of high vol
axes[2].plot(test_dates, test_proba, color="purple", linewidth=1)
axes[2].axhline(0.5, color="gray", linestyle="--")      # 50% decision line
axes[2].set_ylim(0, 1)
axes[2].set_title("Model probability of HIGH volatility next month")
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_test_predictions.png"), dpi=120)
plt.show()      # opens the chart window (close it to let the script continue)

# Chart 2: cross validation scores, model vs naive
fig, ax = plt.subplots(figsize=(8, 4))
x_pos = np.arange(len(cv_results))
ax.bar(x_pos - 0.2, cv_results["balanced_acc"], width=0.4, label="MLP model")
ax.bar(x_pos + 0.2, cv_results["naive_balanced_acc"], width=0.4, label="Naive baseline")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)  # 0.5 = coin flip
ax.set_xticks(x_pos)
ax.set_xticklabels([f"Fold {f}" for f in cv_results["fold"]])
ax.set_ylabel("Balanced accuracy")
ax.set_title("Cross-validation: model vs naive baseline")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cv_scores.png"), dpi=120)
plt.show()
 
print(f"Step 11 done: charts saved in the '{OUTPUT_DIR}' folder")

#12 Predict the regime for the next month
# train one last model on ALL labeled history, then predict using the
# most recent day (saved in Step 7).

live_model = build_model()
live_model.fit(X, y)
 
p_high = live_model.predict_proba(latest_row)[0, 1]    # probability of HIGH vol
prediction = "HIGH" if p_high >= 0.5 else "LOW"
 
print(f"Step 12: as of {latest_row.index[0].date()}")
print(f"  Probability of HIGH volatility in the next {HORIZON} trading days: {p_high:.1%}")
print(f"  Predicted regime: {prediction}")

#13 Final report

cm = confusion_matrix(y_final_test, test_pred)     #same table from step 10
storms_caught = cm[1,1]     #actual high, predicted high
storms_total = cm[1].sum()    # all actual high days
calm_caught = cm[0,0]       #actual low, predicted low
calm_total = cm[0].sum()    # all actual low days

difference = model_test_bal - naive_test_bal    # diff between model and naive

#Verdict and final final report - a diff smaller than one point will be a tie
if difference > 0.01:
    verdict = "BEAT the baseline model"
elif difference >= -0.01:
    verdict = "performed about the SAME as the baseline model"
else:
    verdict = "did NOT beat the simple rule"

if p_high < 0.30:
    outlook = "a CALM month is likely"
elif p_high < 0.50:
    outlook = "leaning CALM, but not very confident"
elif p_high < 0.70:
    outlook = "leaning STORMY, but not very confident"
else:
    outlook = "a STORMY month is likely"

cv_model_avg = cv_results["balanced_acc"].mean()
cv_naive_avg = cv_results["naive_balanced_acc"].mean()

print("\n" + "=" * 60)
print(f"    {TICKER} VOLATILIY REGIME REPORT")
print("=" * 60)

print("\nWHAT THIS MODEL DOES")
print(f" It predicts wethr the next {HORIZON} trading days (about 1 month)")
print(f" of {TICKER} will be CALM (low volatility) or STORMY(high volatility).")

print(f"\nHOW IT DID ON RECENT DATA IT NEVER SAW")
print(f" Period: {X.index[final_test_idx[0]].date()} to {X.index[final_test_idx[-1]].date()}")
print(f" Model score:       {model_test_bal:.1%}")
print(f" Simple rule score: {naive_test_bal:.1%}    ('next month will be like this month')")
print(f" Coin flip:             50%")
print(f" Result: the model {verdict}")

print("\nCALM vs STORMY MONTHS")
print(f" Calm days correctly spotted:   {calm_caught} of {calm_total} ({calm_caught / calm_total:.0%})")
print(f" Stormy days correctly spotted: {storms_caught} of {storms_total} ({storms_caught / storms_total:.0%})")
if storms_caught / storms_total < 0.6:
    print(" Note: The model does a LOT better identifying calm periods andtends to miss stormy periods. Dont rely on it as a warning system")

print(f"\nTESTEDACROSS {len(cv_results)} DIFFERENT TIME PERIODS (2005 onward)")
print(f" Model average:        {cv_model_avg:.1%}")
print(f" Simple rule average:  {cv_naive_avg:.1%}")

print(f"\nOUTLOOK FOR THE NEXT MONTH (as of {latest_row.index[0].date()})")
print(f"  Chance of a stormy month: {p_high:.0%}")
print(f"  In plain words: {outlook}.")

print("\n" + "=" * 60)

print("\nDone. Educational project only - not investment advice lol.")






