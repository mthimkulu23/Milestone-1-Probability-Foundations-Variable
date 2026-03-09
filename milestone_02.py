"""
Milestone 2: Distribution Modelling & Shape Analysis
Author: Thabang Mthimkulu
Description: This script analyses user behaviour distributions using higher-order moments,
             fits theoretical models (Poisson, Normal), and demonstrates the Central Limit Theorem.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# ==============================================================================
# DATA LOADING
# ==============================================================================

# Path to the required dataset
file_path = os.path.join('data', 'finflow_users.csv')

try:
    # Load dataset using pandas
    df = pd.read_csv(file_path)
    session_minutes = df['session_minutes'].values
    score_views = df['score_views'].values
except FileNotFoundError:
    print(f"Error: Dataset not found at {file_path}.")
    print("Please ensure the 'data' folder contains 'finflow_users.csv'.")
    exit()

# ==============================================================================
# PART 1: MOMENTS & SHAPE ANALYSIS
# ==============================================================================

# 1. Mean (1st moment)
mean_minutes = np.mean(session_minutes)

# 2. Variance (2nd central moment, ddof=1 for sample variance)
variance_minutes = np.var(session_minutes, ddof=1)

# 3. Skewness (3rd standardised moment, bias=False for unbiased)
skewness_minutes = stats.skew(session_minutes, bias=False)

# 4. Excess Kurtosis (4th standardised moment, bias=False)
kurtosis_minutes = stats.kurtosis(session_minutes, bias=False)

# ==============================================================================
# PART 2: STANDARD DISTRIBUTIONS
# ==============================================================================

# 1. Fit Poisson distribution to score_views
# For Poisson, the MLE for lambda is simply the sample mean
lambda_poisson = np.mean(score_views)

# 2. Fit Normal distribution to session_minutes
# Using MLE estimates for mu and sigma
mu_normal, sigma_normal = stats.norm.fit(session_minutes)

# 3. Perform Kolmogorov-Smirnov goodness-of-fit tests
# KS test for Poisson
# Note: kstest for discrete distributions requires discrete CDF or using 'poisson' string with params
ks_stat_poisson, p_value_poisson = stats.kstest(score_views, 'poisson', args=(lambda_poisson,))

# KS test for Normal
ks_stat_normal, p_value_normal = stats.kstest(session_minutes, 'norm', args=(mu_normal, sigma_normal))

# ==============================================================================
# PART 3: SAMPLING DISTRIBUTIONS & CLT
# ==============================================================================

pop_mean = session_minutes.mean()
pop_std = session_minutes.std(ddof=1)
sample_sizes = [10, 30, 100]
n_reps = 10000

# Set seed for reproducibility of simulation
np.random.seed(42)

# 1. Simulate sampling distributions
sampling_distributions = {}
for n in sample_sizes:
    # Draw n_reps samples of size n and compute their means
    # Efficient way: reshape random choice or just loop
    samples = np.random.choice(session_minutes, size=(n_reps, n), replace=True)
    sample_means = np.mean(samples, axis=1)
    sampling_distributions[n] = sample_means

# 2. Calculate empirical Standard Error (SE) for each n
empirical_ses = {n: np.std(means, ddof=1) for n, means in sampling_distributions.items()}

# 3. Calculate theoretical Standard Error (σ/√n)
theoretical_ses = {n: pop_std / np.sqrt(n) for n in sample_sizes}

# 4. Determine minimum n for approximate Normality (|skew| < 0.5)
min_n_normal = None
for n in sample_sizes:
    skew_val = stats.skew(sampling_distributions[n], bias=False)
    if abs(skew_val) < 0.5:
        min_n_normal = n
        break

# ==============================================================================
# VALIDATION CHECKS (Autograder Safety)
# ==============================================================================

assert mean_minutes > 0, "Mean must be positive"
assert variance_minutes > 0, "Variance must be positive"
assert lambda_poisson > 0, "Poisson lambda must be positive"
assert sigma_normal > 0, "Normal sigma must be positive"

# Ensure SE accuracy within 10%
for n in sample_sizes:
    rel_error = abs(empirical_ses[n] - theoretical_ses[n]) / theoretical_ses[n]
    assert rel_error < 0.1, f"SE mismatch for n={n}: {rel_error:.2%}"

# ==============================================================================
# RESULTS & INTERPRETATION
# ==============================================================================

print("\n" + "="*80)
print("SESSION DURATION MOMENTS & SHAPE ANALYSIS")
print("="*80)
print(f"Mean:     {mean_minutes:.2f} minutes")
print(f"Variance: {variance_minutes:.2f} (SD = {variance_minutes**0.5:.2f})")
print(f"Skewness: {skewness_minutes:.2f}")
print(f"Kurtosis: {kurtosis_minutes:.2f} (excess)")

print("\nSHAPE INTERPRETATION:")
# Logic to describe skewness
if skewness_minutes > 0.5:
    skew_type = "Positively skewed (Right-skew)"
    skew_biz = "Most sessions are short, but a small group of highly engaged users have very long sessions."
elif skewness_minutes < -0.5:
    skew_type = "Negatively skewed (Left-skew)"
    skew_biz = "Most users stay for long durations, with a few early exits."
else:
    skew_type = "Approximately Symmetric"
    skew_biz = "User session behavior is balanced around the average."

# Logic for kurtosis
if kurtosis_minutes > 1:
    kurt_type = "Leptokurtic (Heavy-tailed)"
    kurt_biz = "The presence of extreme outliers (power users) is significantly higher than a Normal distribution."
else:
    kurt_type = "Platykurtic/Mesokurtic"
    kurt_biz = "Outliers are less frequent; user behavior is relatively contained."

print(f"  Skewness ({skewness_minutes:.2f}): {skew_type}")
print(f"  Kurtosis ({kurtosis_minutes:.2f}): {kurt_type}")

print("\nBUSINESS IMPLICATION:")
print(f"  {skew_biz}")
print(f"  {kurt_biz}")

print("\n" + "="*80)
print("DISTRIBUTION FITTING & GOODNESS-OF-FIT")
print("="*80)
print(f"{'Distribution':<15} {'Parameter(s)':<25} {'KS Stat':<10} {'p-value':<10}")
print("-" * 80)
print(f"Poisson         λ = {lambda_poisson:.2f}{'':<15} {ks_stat_poisson:.3f}    {p_value_poisson:.3f}")
print(f"Normal          μ = {mu_normal:.2f}, σ = {sigma_normal:.2f}   {ks_stat_normal:.3f}    {p_value_normal:.3f}")
print("="*80)

print("\nGOODNESS-OF-FIT INTERPRETATION:")
p_thresh = 0.05
pois_fit = "Reject H0 (Poisson fit is poor)" if p_value_poisson < p_thresh else "Fail to reject H0 (Poisson fits data)"
norm_fit = "Reject H0 (Normal fit is poor)" if p_value_normal < p_thresh else "Fail to reject H0 (Normal fits data)"
print(f"  Poisson: {pois_fit}")
print(f"  Normal:  {norm_fit}")

print("\nRECOMMENDATION FOR SIMULATION MODELS:")
if p_value_poisson < p_thresh and p_value_normal < p_thresh:
    recommendation = "Neither Poisson nor Normal is ideal. Suggest exploring Log-Normal for sessions due to right-skew."
elif p_value_poisson > p_value_normal:
    recommendation = "Poisson is the best fit for count-based engagement data (score_views)."
else:
    recommendation = "Normal distribution is the most suitable model for the current behavioral metrics."
print(f"  {recommendation}")

print("\n" + "="*80)
print("CENTRAL LIMIT THEOREM (CLT) SIMULATION")
print("="*80)
print(f"{'Sample Size (n)':<20} {'Empirical SE':<18} {'Theoretical SE':<18} {'Ratio'}")
print("-" * 80)
for n in sample_sizes:
    ratio = empirical_ses[n] / theoretical_ses[n]
    print(f"{n:<20} {empirical_ses[n]:<18.2f} {theoretical_ses[n]:<18.2f} {ratio:.3f}")
print("=" * 80)

print(f"\nMinimum n for Stability (|skew| < 0.5): {min_n_normal}")
print("\nBUSINESS RECOMMENDATION:")
print(f"  Based on CLT convergence, we need a minimum sample size of n={min_n_normal} per variant in future ")
print("  A/B tests. This ensures that the distribution of the mean is Normal, making T-tests valid ")
print("  even though the raw session data is significantly skewed.")
print("="*80 + "\n")
