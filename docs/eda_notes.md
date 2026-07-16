# EDA Notes — Loan Default Prediction (Give Me Some Credit)

## 1. Load data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/raw/cs-training.csv')
df = df.drop(columns=['Unnamed: 0'])
df.shape  # (150000, 11)
```

## 2. Target / class balance
```python
df['SeriousDlqin2yrs'].value_counts(normalize=True)
```
**Finding:** 93.3% no-default / 6.7% default → imbalanced. Accuracy alone is misleading; use precision/recall/ROC-AUC later.

## 3. Missing values
```python
df.isnull().sum()
```
**Finding:**
- `MonthlyIncome` → 19.8% missing (29,731 rows)
- `NumberOfDependents` → 2.6% missing (3,924 rows)

## 4. Summary stats / outlier scan
```python
df.describe()
```
**Findings:**
- `age`: min = 0 → 1 invalid row, drop it
- `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`: max = 98 → placeholder/error codes, not real counts (269 rows affected)
- `RevolvingUtilizationOfUnsecuredLines`: max = 50,708 → should be a ratio, impossible value (241 rows > 10)
- `DebtRatio`: max = 329,664 → impossible ratio (28,877 rows > 10)

## 5. Root cause: DebtRatio outliers ≈ missing income
```python
extreme_debtratio = df[df['DebtRatio'] > 10]
extreme_debtratio['MonthlyIncome'].isnull().sum()   # 26,771 / 28,877 (92.7%)

df[df['MonthlyIncome'].isnull()]['DebtRatio'].describe()
```
**Finding:** Extreme DebtRatio isn't a separate bug — it's a symptom of missing `MonthlyIncome`. Fix income first, then re-check DebtRatio.

## 6. Target vs. features (does the data have signal?)
```python
df.groupby('SeriousDlqin2yrs')[['age','RevolvingUtilizationOfUnsecuredLines',
                                   'NumberOfTimes90DaysLate','DebtRatio']].median()

sns.boxplot(data=df, x='SeriousDlqin2yrs', y='age')
```
**Findings (median):**
| Feature | No default (0) | Default (1) |
|---|---|---|
| age | 52 | 45 |
| RevolvingUtilization | 0.13 | 0.84 |
| DebtRatio | 0.36 | 0.43 |

**Lesson:** Median hid the signal in `NumberOfTimes90DaysLate` (0 vs 0) because most people of both classes have zero. Mean revealed it instead:
```python
df[df['NumberOfTimes90DaysLate'] < 96].groupby('SeriousDlqin2yrs')['NumberOfTimes90DaysLate'].mean()
```
→ 0.050 (no default) vs 0.665 (default), ~13x gap. **Rule of thumb: check both median and mean — median hides tail signal, mean can be distorted by junk values, so clean first.**

## Key takeaways for Step 4 (Cleaning)
1. Drop the 1 row with age = 0
2. Decide fix for 96/98 placeholder codes (3 columns, 269 rows)
3. Impute `MonthlyIncome` (19.8% missing) and `NumberOfDependents` (2.6% missing)
4. Re-check `DebtRatio` after income is imputed
5. Cap/handle extreme `RevolvingUtilizationOfUnsecuredLines`
6. **Split train/val/test BEFORE computing any fill values** (medians etc.) to avoid data leakage — compute stats on train only, apply to val/test
