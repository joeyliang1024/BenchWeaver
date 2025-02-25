import numpy as np
import pandas as pd
from scipy.stats import entropy, chi2_contingency
import warnings

def compute_difficulty_distribution(df, difficulty_col='difficulty'):
    counts = df[difficulty_col].value_counts().sort_index()
    total = counts.sum()
    return counts / total, counts

def compute_kl_divergence(pA, pB):
    return entropy(pA, pB)

def compute_js_divergence(pA, pB):
    M = 0.5 * (pA + pB)
    return 0.5 * (entropy(pA, M) + entropy(pB, M))

def compute_chi_square_test(N_A, N_B):
    chi2, p_value, _, _ = chi2_contingency([N_A, N_B])
    return chi2, p_value

def compute_wasserstein_distance(cdf_A, cdf_B):
    return np.sum(np.abs(cdf_A - cdf_B))

def balance_datasets(df_A: pd.DataFrame, df_B: pd.DataFrame, difficulty_col='difficulty'):
    pA, N_A = compute_difficulty_distribution(df_A, difficulty_col)
    pB, N_B = compute_difficulty_distribution(df_B, difficulty_col)
    
    difficulty_levels = sorted(set(df_A[difficulty_col]).union(set(df_B[difficulty_col])))
    pA = pA.reindex(difficulty_levels, fill_value=0).values
    pB = pB.reindex(difficulty_levels, fill_value=0).values
    N_A = N_A.reindex(difficulty_levels, fill_value=0).values
    N_B = N_B.reindex(difficulty_levels, fill_value=0).values

    print("Before balancing:")
    print(f"KL: {compute_kl_divergence(pA, pB):.4f}")
    print(f"JS: {compute_js_divergence(pA, pB):.4f}")
    chi2, p_value = compute_chi_square_test(N_A, N_B)
    wass = compute_wasserstein_distance(np.cumsum(pA), np.cumsum(pB))
    print(f"Chi-Square: {chi2:.4f}, p-value: {p_value:.4f}, Wasserstein: {wass:.4f}")
    print("=====================================================")
    p_target = (pA + pB) / 2
    N_target = np.round(min(len(df_A), len(df_B)) * p_target).astype(int)
    d_levels = {k: i for i, k in enumerate(difficulty_levels)}
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    resampled_A = df_A.groupby(difficulty_col, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), N_target[d_levels[x.name]]), random_state=42)
    )
    resampled_B = df_B.groupby(difficulty_col, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), N_target[d_levels[x.name]]), random_state=42)
    )

    # Compute again after balancing
    pA_bal, N_A_bal = compute_difficulty_distribution(resampled_A, difficulty_col)
    pB_bal, N_B_bal = compute_difficulty_distribution(resampled_B, difficulty_col)
    pA_bal = pA_bal.reindex(difficulty_levels, fill_value=0).values
    pB_bal = pB_bal.reindex(difficulty_levels, fill_value=0).values
    N_A_bal = N_A_bal.reindex(difficulty_levels, fill_value=0).values
    N_B_bal = N_B_bal.reindex(difficulty_levels, fill_value=0).values

    print("After balancing:")
    print(f"KL: {compute_kl_divergence(pA_bal, pB_bal):.4f}")
    print(f"JS: {compute_js_divergence(pA_bal, pB_bal):.4f}")
    chi2_bal, p_value_bal = compute_chi_square_test(N_A_bal, N_B_bal)
    wass_bal = compute_wasserstein_distance(np.cumsum(pA_bal), np.cumsum(pB_bal))
    print(f"Chi-Square: {chi2_bal:.4f}, p-value: {p_value_bal:.4f}, Wasserstein: {wass_bal:.4f}")
    print("=====================================================")
    
    return resampled_A, resampled_B

if __name__ == "__main__":
    df_A = pd.DataFrame({
        'question': np.random.choice(['q1', 'q2', 'q3', 'q4'], 1000),
        'answer': np.random.choice(['a1', 'a2', 'a3', 'a4'], 1000),
        'difficulty': np.random.choice(['easy', 'normal', 'hard'], 1000),
    })
    df_B = pd.DataFrame({
        'question': np.random.choice(['q1', 'q2', 'q3', 'q4'], 1200),
        'answer': np.random.choice(['a1', 'a2', 'a3', 'a4'], 1200),
        'difficulty': np.random.choice(['easy', 'normal', 'hard'], 1200),
    })
    print("Before balancing A:")
    print(df_A['difficulty'].value_counts())
    print("\nBefore balancing B:")
    print(df_B['difficulty'].value_counts())
    print("=====================================================")
    balanced_A, balanced_B = balance_datasets(df_A, df_B)
    print("Sample counts in balanced A:")
    print(balanced_A['difficulty'].value_counts())
    print("\nSample counts in balanced B:")
    print(balanced_B['difficulty'].value_counts())