# AegisFin-AI Phase 1 — Recovered Best Results

## Dataset
- Rows: 307,511
- Original columns: 122
- TARGET=0: 282,686 (91.9271%)
- TARGET=1: 24,825 (8.0729%)
- Majority/minority ratio: 11.39:1
- Duplicate rows: 0
- Constant columns: 0

## Feature engineering
13 engineered features were added before splitting:
- DAYS_EMPLOYED_ANOMALY
- AGE_YEARS
- EMPLOYMENT_YEARS
- CREDIT_INCOME_RATIO
- ANNUITY_INCOME_RATIO
- CREDIT_ANNUITY_RATIO
- GOODS_CREDIT_RATIO
- INCOME_PER_FAMILY_MEMBER
- INCOME_PER_CHILD
- EXT_SOURCE_MEAN
- EXT_SOURCE_STD
- EXT_SOURCE_MIN
- EXT_SOURCE_MAX

Feature matrix after leakage-safe preprocessing:
- 183 features
- Identifier SK_ID_CURR removed
- PCA not used
- Feature reduction not selected

## Feature-reduction experiment
Quick validation LightGBM:
- All features PR-AUC: 0.258849
- Reduced features PR-AUC: 0.254380
- All features ROC-AUC: 0.766208
- Reduced features ROC-AUC: 0.765900

Therefore the previous notebook selected the all-feature representation.

## Imbalance
Selected strategy:
- sqrt_weight
- scale_pos_weight = 3.374571993782351
- No RandomOverSampler in the final training path

## Optuna results
Best PR-AUC from the completed 3-fold Optuna studies:

### XGBoost
Best PR-AUC: 0.24049957357060683

- n_estimators: 800
- max_depth: 6
- learning_rate: 0.011014097143819117
- min_child_weight: 7.068239480981438
- subsample: 0.7096834432905521
- colsample_bytree: 0.6727680575448478
- gamma: 4.7444276862666666
- reg_alpha: 6.732248920775331
- reg_lambda: 4.661695418105682

Previous final-refit preparation:
- best XGBoost estimators: 797

### LightGBM
Best PR-AUC: 0.2367309855541407

- n_estimators: 1100
- num_leaves: 30
- max_depth: 5
- learning_rate: 0.021421886418348784
- min_child_samples: 102
- min_split_gain: 0.7851759613930136
- subsample: 0.719885823755426
- colsample_bytree: 0.8299820534447641
- reg_alpha: 0.09163741808778776
- reg_lambda: 0.014234237430895474

Previous final-refit preparation:
- best LightGBM estimators: 1084

### CatBoost
Best PR-AUC: 0.2396219077142868

- iterations: 1200
- depth: 4
- learning_rate: 0.035372656148130316
- l2_leaf_reg: 1.6666983286066417
- random_strength: 0.19515477895583855
- bagging_temperature: 4.7444276862666666
- border_count: 248

Previous final-refit preparation:
- best CatBoost iterations: 1200

## Fixed rerun decision
The completed Optuna search has the highest PR-AUC for XGBoost (0.2404995736), so the new fast rerun notebook trains only XGBoost using the recovered configuration.

## Important limitation
The uploaded notebook did not contain a completed printed `model_results_df` / final test-metrics table. Therefore this summary does NOT invent a final validation or test score for the selected XGBoost model.

The new notebook performs one fixed XGBoost training run and one untouched-test evaluation, then saves the real metrics.
