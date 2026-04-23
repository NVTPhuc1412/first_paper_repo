import pandas as pd

# 1. Load PA results to find the best epoch for each model
df_pa = pd.concat([
    pd.read_csv('./test_results.csv'),
    pd.read_csv('./baseline_test_results.csv')
])
df_pa['model'] = df_pa['encoder'].fillna('None') + '+' + df_pa['detector']
df_pa['epoch'] = df_pa['epoch'].fillna(0)

# Group by model and epoch, mean metrics in PA results
grouped_pa = df_pa.groupby(['model', 'epoch']).mean(numeric_only=True).reset_index()

# Select the best epoch for each model based on the highest PA-F1 score
best_results_pa = grouped_pa.loc[grouped_pa.groupby('model')['f1'].idxmax()]

df_rigorous = pd.merge(
    df_pa,
    best_results_pa[['model', 'epoch']],
    on=['model', 'epoch']
)

score_by_diff = df_rigorous.groupby(['model', 'test_set']).mean(numeric_only=True)['auc'].unstack(level=-1)
cols = [
    "Synthetic_Easy",
    "Synthetic_Medium",
    "Synthetic_Hard",
    "Real_Easy",
    "Real_Medium",
    "Real_Hard",
]
print(score_by_diff[cols].to_latex())