"""
Milestone 3: Statistical Inference & Hypothesis Testing
Author: Thabang Mthimkulu
Description: This script performs statistical inference using confidence intervals,
             hypothesis testing (t-test, chi-square), and A/B test analysis with 
             multiple comparison corrections.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
import os

# ==============================================================================
# DATA LOADING
# ==============================================================================

try:
    df = pd.read_csv('data/finflow_users.csv')
    ab_df = pd.read_csv('data/finflow_ab_test.csv')
except FileNotFoundError as e:
    print(f"Error: Required dataset file not found. {e}")
    exit()

# ==============================================================================
# PART 1: CONFIDENCE INTERVALS
# ==============================================================================

# 1. 95% t-based CI for mean session duration
n = len(df)
mean_minutes = df['session_minutes'].mean()
sd_minutes = df['session_minutes'].std(ddof=1)
se_minutes = sd_minutes / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
ci_mean_lower = mean_minutes - t_crit * se_minutes
ci_mean_upper = mean_minutes + t_crit * se_minutes
margin_error_mean = t_crit * se_minutes

# 2. 95% Wilson score interval for premium conversion proportion
successes = df['premium_user'].sum()
n_total = len(df)
ci_prop_lower, ci_prop_upper = proportion_confint(successes, n_total, alpha=0.05, method='wilson')
p_hat = successes / n_total
margin_error_prop = (ci_prop_upper - ci_prop_lower) / 2

# ==============================================================================
# PART 2: BOOTSTRAP METHODS
# ==============================================================================

session_minutes = df['session_minutes'].values
n_boot = 10000
np.random.seed(42)

# Bootstrap resampling for median
bootstrap_medians = np.zeros(n_boot)
for i in range(n_boot):
    # Draw sample with replacement
    sample = np.random.choice(session_minutes, size=len(session_minutes), replace=True)
    bootstrap_medians[i] = np.median(sample)

# 95% percentile CI for bootstrap median
ci_boot_lower = np.percentile(bootstrap_medians, 2.5)
ci_boot_upper = np.percentile(bootstrap_medians, 97.5)
point_estimate_median = np.median(session_minutes)

# ==============================================================================
# PART 3: HYPOTHESIS TESTING I (T-TEST)
# ==============================================================================

# Split by premium status
free_users = df[df['premium_user'] == 0]['session_minutes']
premium_users = df[df['premium_user'] == 1]['session_minutes']

h0_ttest = "H₀: μ_premium ≤ μ_free (Premium users do not have longer sessions than free users)"
ha_ttest = "Hₐ: μ_premium > μ_free (Premium users have significantly longer sessions)"

# Assumption Check: Normality (Shapiro-Wilk)
# Note: For large N, Shapiro-Wilk is very sensitive. We report it as required.
shapiro_free_stat, shapiro_free_p = stats.shapiro(free_users)
shapiro_premium_stat, shapiro_premium_p = stats.shapiro(premium_users)
normality_ok = (shapiro_free_p > 0.05) and (shapiro_premium_p > 0.05)

# Assumption Check: Equal Variance (Levene's test)
levene_stat, levene_p = stats.levene(free_users, premium_users)
equal_var = levene_p > 0.05

# Perform one-tailed Welch's t-test (equal_var=False handles non-pooled variance)
t_stat, p_two_tail = stats.ttest_ind(premium_users, free_users, equal_var=False)
p_value_ttest = p_two_tail / 2 if t_stat > 0 else 1 - p_two_tail/2
reject_h0_ttest = p_value_ttest < 0.05

# Effect Size: Cohen's d (pooled SD)
n1, n2 = len(free_users), len(premium_users)
sd1, sd2 = free_users.std(ddof=1), premium_users.std(ddof=1)
pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1 + n2 - 2))
cohens_d = (premium_users.mean() - free_users.mean()) / pooled_sd

# Sample size needed for 80% power (approximate)
# Z_beta = 0.84, Z_alpha = 1.645 (one-tailed 0.05)
z_alpha = 1.645
z_beta = 0.84
n_needed_ttest = 2 * ((z_alpha + z_beta)**2 * pooled_sd**2) / (premium_users.mean() - free_users.mean())**2

# ==============================================================================
# PART 4: HYPOTHESIS TESTING II (CHI-SQUARE)
# ==============================================================================

# Create contingency table
contingency_table = pd.crosstab(df['risk_profile'], df['premium_user'])

# Perform chi-square test of independence
chi2_stat, p_value_chi2, dof_chi2, expected = stats.chi2_contingency(contingency_table)

# Check expected count assumption
min_expected = expected.min()
assumption_met = min_expected >= 5 or (np.sum(expected >= 5) / expected.size >= 0.8 and min_expected >= 1)

# Calculate Cramér's V effect size
n_chi2 = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2_stat / (n_chi2 * min(contingency_table.shape[0]-1, contingency_table.shape[1]-1)))

# ==============================================================================
# PART 5: MULTIPLE COMPARISONS (A/B TEST)
# ==============================================================================

# Get conversion rates by variant
variant_groups = ab_df.groupby('variant')['converted']
conversion_rates = variant_groups.mean()
counts = variant_groups.sum()
nobs = variant_groups.count()

control_rate = conversion_rates['control']
control_success = counts['control']
control_n = nobs['control']

results = []
variants_to_test = [v for v in ab_df['variant'].unique() if v != 'control']
alpha = 0.05
m = len(variants_to_test)
alpha_adj = alpha / m

for variant in variants_to_test:
    v_success = counts[variant]
    v_n = nobs[variant]
    v_rate = conversion_rates[variant]
    
    # Run two-proportion z-test (one-tailed 'larger')
    stat, p_val = proportions_ztest([v_success, control_success], [v_n, control_n], alternative='larger')
    
    significant = p_val < alpha_adj
    abs_lift = v_rate - control_rate
    rel_lift = (v_rate - control_rate) / control_rate
    
    results.append({
        'variant': variant,
        'conversion_rate': v_rate,
        'p_value': p_val,
        'significant': significant,
        'abs_lift': abs_lift,
        'rel_lift': rel_lift
    })

results_df = pd.DataFrame(results)

# ==============================================================================
# VALIDATION CHECKS
# ==============================================================================

assert ci_mean_lower < mean_minutes < ci_mean_upper, "Mean CI must contain point estimate"
assert ci_prop_lower < p_hat < ci_prop_upper, "Proportion CI must contain point estimate"
assert ci_boot_lower < point_estimate_median < ci_boot_upper, "Bootstrap CI must contain median"
assert cohens_d > 0, "Cohen's d should be positive (premium > free)"
assert 0 <= p_value_chi2 <= 1, "p-value must be between 0 and 1"
assert len(results_df) == 4, "Must test matched variants"

# ==============================================================================
# OUTPUT & INTERPRETATION
# ==============================================================================

print("\n" + "="*80)
print("CONFIDENCE INTERVALS (95%)")
print("="*80)
print(f"{'Metric':<25} {'Point Estimate':<20} {'95% CI Bounds':<25} {'MoE'}")
print("-"*80)
print(f"{'Mean Duration (min)':<25} {mean_minutes:<20.2f} [{ci_mean_lower:.2f}, {ci_mean_upper:.2f}] {'':<5} ±{margin_error_mean:.2f}")
print(f"{'Conversion Rate (%)':<25} {p_hat:<20.2%} [{ci_prop_lower:.2%}, {ci_prop_upper:.2%}] {'':<5} ±{margin_error_prop:.2%}")
print("="*80)

print("\nBUSINESS INTERPRETATION:")
print(f"  We are 95% confident that the true average session length is between {ci_mean_lower:.1f} and {ci_mean_upper:.1f} minutes.")
print(f"  The worst-case scenario for conversion is {ci_prop_lower:.1%}, which helps in conservative budgeting.")

print("\n" + "="*80)
print("BOOTSTRAP VS PARAMETRIC CONFIDENCE INTERVALS")
print("="*80)
boot_width = ci_boot_upper - ci_boot_lower
param_width = ci_mean_upper - ci_mean_lower
print(f"{'Statistic':<15} {'Estimate':<15} {'95% CI Width':<20} {'Relative Width'}")
print("-"*80)
print(f"{'Median (boot)':<15} {point_estimate_median:<15.1f} {boot_width:<20.2f} 1.00x")
print(f"{'Mean (param)':<15} {mean_minutes:<15.1f} {param_width:<20.2f} {param_width/boot_width:.2f}x")
print("="*80)

print("\nBOOTSTRAP INTERPRETATION:")
print(f"  The bootstrap median ({point_estimate_median:.1f}) provides a robust estimate unaffected by 'power user' outliers.")
print("  Recommendation: Use the median for typical user experience metrics, as it is more stable than the mean.")

print("\n" + "="*80)
print("HYPOTHESIS TEST: PREMIUM VS FREE USER ENGAGEMENT")
print("="*80)
print(f"H0: {h0_ttest}")
print(f"Ha: {ha_ttest}")
print(f"\nAssumption Checks:")
print(f"  Normality (p): Free={shapiro_free_p:.4f}, Premium={shapiro_premium_p:.4f} → {'OK' if normality_ok else 'VIOLATED (N large enough for CLT)'}")
print(f"  Equal Variance: p={levene_p:.4f} → {'OK' if equal_var else 'VIOLATED (Using Welch)'}")
print(f"\nTest Results (Welch t-test):")
print(f"  t = {t_stat:.2f}, p = {p_value_ttest:.4f}")
print(f"  Decision: {'REJECT H0' if reject_h0_ttest else 'FAIL TO REJECT H0'}")
print(f"\nEffect Size:")
print(f"  Cohen's d = {cohens_d:.2f} ({'small' if abs(cohens_d)<0.2 else 'medium' if abs(cohens_d)<0.5 else 'large'} effect)")
print(f"  Approx. n needed per group for 80% power: {n_needed_ttest:.0f}")
print("="*80)

print("\nBUSINESS INTERPRETATION:")
if reject_h0_ttest:
    print("  Premium status is significantly associated with higher session durations. ")
    print("  This suggests premium features or the premium user segment shows deeper engagement patterns.")
else:
    print("  We found no significant evidence that premium users engage longer than free users.")

print("\n" + "="*80)
print("CHI-SQUARE TEST: RISK PROFILE VS CONVERSION")
print("="*80)
print(f"Chi-square Stat: {chi2_stat:.2f}, p-value: {p_value_chi2:.4f}")
print(f"Effect Size (Cramér's V): {cramers_v:.3f}")
print(f"Assumption Check (Expected counts >= 5): {'OK' if assumption_met else 'VIOLATED'}")
print("="*80)

print("\nBUSINESS INTERPRETATION:")
if p_value_chi2 < 0.05:
    print(f"  Risk profile is significantly associated with premium status (V={cramers_v:.2f}).")
    print("  Marketing targeting should be tailored based on a user's risk stance.")
else:
    print("  No significant association found between risk profile and premium conversion.")

print("\n" + "="*80)
print("A/B TEST ANALYSIS (BONFERRONI CORRECTION)")
print("="*80)
print(f"Control Rate: {control_rate:.1%}")
print(f"Adjusted Alpha (Bonferroni): {alpha_adj:.4f}\n")
print(results_df.to_string(index=False, formatters={
    'conversion_rate': '{:.1%}'.format, 'p_value': '{:.4f}'.format,
    'abs_lift': '{:+.2%}'.format, 'rel_lift': '{:+.1%}'.format
}))
print("="*80)

significant_variants = results_df[results_df['significant']]
if not significant_variants.empty:
    best = significant_variants.loc[significant_variants['abs_lift'].idxmax()]
    print(f"\nDEPLOYMENT RECOMMENDATION: Deploy Variant {best['variant'].upper()}")
    print(f"  This variant achieved a significant {best['rel_lift']:+.1%} relative lift over control.")
    print("  The result survives Bonferroni correction, ensuring we are not deploying a false positive.")
else:
    print("\nRECOMMENDATION: Do not deploy any variant.")
    print("  No variants showed statistically significant improvement after Bonferroni correction.")
print("="*80 + "\n")
