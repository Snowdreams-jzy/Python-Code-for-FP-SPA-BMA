import pandas as pd
import numpy as np
from scipy.stats import norm

np.random.seed(12)
df = pd.read_excel('data.xlsx')

target_col = 'ref'
base_models = ['XGB', 'CNN', 'RF', 'SVM']
partition_col = 'GWC'

train_df = df.iloc[:71].copy()
test_df = df.iloc[71:].copy()


def calculate_spa_weights(data, sigma_ref):
    n = len(data)
    if n == 0: return np.array([0.25] * 4)
    epsilon, delta = 0.1 * sigma_ref, 0.5 * sigma_ref
    mu_scores = []
    for m in base_models:
        residuals = np.abs(data[target_col] - data[m])
        a = np.sum(residuals <= epsilon) / n
        c = np.sum(residuals > delta) / n
        mu = a - c
        mu_scores.append(max(0, mu))

    total_mu = sum(mu_scores)
    return np.array(mu_scores) / total_mu if total_mu > 0 else np.array([0.25] * 4)


def run_bma_em(f_matrix, y_obs, w_prior, max_iter=100, tol=1e-4):
    n_samples, n_models = f_matrix.shape
    W = w_prior.copy()
    sigma2 = np.var(y_obs)

    for _ in range(max_iter):
        prev_W = W.copy()
        probs = np.zeros((n_samples, n_models))
        for m in range(n_models):
            probs[:, m] = W[m] * norm.pdf(y_obs, loc=f_matrix[:, m], scale=np.sqrt(sigma2))
        z = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)

        W = (z.mean(axis=0) * w_prior) / np.sum(z.mean(axis=0) * w_prior)

        diff = 0
        for m in range(n_models):
            diff += np.sum(z[:, m] * (y_obs - f_matrix[:, m]) ** 2)
        sigma2 = diff / n_samples

        if np.linalg.norm(W - prev_W) < tol: break
    return W


def get_bma_prediction(f_vals, W):
    return np.sum(W * f_vals)

# Ablation studies
y_std_train = train_df[target_col].std()

# FP-SPA-BMA
low_th = train_df[partition_col].quantile(0.3)
high_th = train_df[partition_col].quantile(0.7)
train_df['Zone'] = train_df[partition_col].apply(lambda x: 1 if x < low_th else (3 if x >= high_th else 2))
test_df['Zone'] = test_df[partition_col].apply(lambda x: 1 if x < low_th else (3 if x >= high_th else 2))

m1_weights = {}
for z in [1, 2, 3]:
    z_df = train_df[train_df['Zone'] == z]
    w_spa = calculate_spa_weights(z_df, y_std_train)
    W_post = run_bma_em(z_df[base_models].values, z_df[target_col].values, w_spa)
    m1_weights[z] = W_post

# SPA-BMA
w_spa_all = calculate_spa_weights(train_df, y_std_train)
W_m2 = run_bma_em(train_df[base_models].values, train_df[target_col].values, w_spa_all)

# BMA
w_equal = np.array([0.25] * 4)
W_m3 = run_bma_em(train_df[base_models].values, train_df[target_col].values, w_equal)

results = []
for _, row in test_df.iterrows():
    f = row[base_models].values

    p1 = get_bma_prediction(f, m1_weights[row['Zone']])
    p2 = get_bma_prediction(f, W_m2)
    p3 = get_bma_prediction(f, W_m3)

    results.append([row['number'], row['ref'], row['Zone'], p1, p2, p3])

cols = ['number', 'Observed', 'Zone', 'FP_SPA_BMA_Pred', 'SPA_BMA_Pred', 'BMA_Pred']
output_df = pd.DataFrame(results, columns=cols)
output_df.to_excel('Ablation_Study_Predictions.xlsx', index=False)
print("Ablation study prediction results have been exported to 'Ablation_Study_Predictions.xlsx'")