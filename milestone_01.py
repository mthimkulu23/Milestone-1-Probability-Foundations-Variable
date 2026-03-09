"""
Milestone 1: Probability Foundations & Variable Types
Author: [Thabang Mthimkulu]
Description: This script calculates basic and conditional probabilities, verifies Bayes' Theorem,
             and classifies random variables using financial user engagement data.
"""

import pandas as pd
import numpy as np
import os

# ==============================================================================
# DATA LOADING
# ==============================================================================

# Path to the required dataset
file_path = os.path.join('data', 'finflow_users.csv')

try:
    # Load dataset using pandas
    df = pd.read_csv(file_path)
except FileNotFoundError:
    # Handle missing dataset with a clear error message as per requirements
    print(f"Error: Dataset not found at {file_path}.")
    print("Please ensure the 'data' folder contains 'finflow_users.csv'.")
    exit()

# ==============================================================================
# PART 1: SAMPLE SPACES & BASIC PROBABILITY
# ==============================================================================

# 1. Define sample space for premium_user (Binary set)
sample_space_premium = set(df['premium_user'].unique())

# 2. Calculate P(premium_user = 1) - Baseline conversion
p_premium = (df['premium_user'] == 1).mean()

# 3. Calculate P(score_views >= 5) - High engagement
p_high_engagement = (df['score_views'] >= 5).mean()

# 4. Calculate P(risk_profile = 'aggressive') - Risk preference
p_aggressive = (df['risk_profile'] == 'aggressive').mean()

# 5. Calculate joint probability P(score_views >= 5 AND premium_user = 1)
# Uses vectorized boolean AND operation
p_joint = ((df['score_views'] >= 5) & (df['premium_user'] == 1)).mean()

# ==============================================================================
# PART 2: CONDITIONAL PROBABILITY & BAYES
# ==============================================================================

# 1. Calculate P(premium = 1 | score_views >= 3)
# Filter users with at least 3 views, then calculate their conversion rate
engaged_mask = df['score_views'] >= 3
p_premium_given_engaged = df[engaged_mask]['premium_user'].mean()

# 2. Calculate P(score_views >= 3 | premium = 1)
# Filter for premium users, then calculate their engagement rate
premium_mask = df['premium_user'] == 1
p_engaged_given_premium = df[premium_mask]['score_views'].ge(3).mean()

# 3. Calculate P(score_views >= 3) - Marginal probability of engagement
p_engaged = engaged_mask.mean()

# 4. Verify Bayes' theorem
# Formula: P(A|B) = [P(B|A) * P(A)] / P(B)
bayes_check = (p_engaged_given_premium * p_premium) / p_engaged

# 5. Calculate odds ratio
# Measures how much engagement increases the odds of being premium
odds_engaged = p_premium_given_engaged / (1 - p_premium_given_engaged)
odds_baseline = p_premium / (1 - p_premium)
odds_ratio = odds_engaged / odds_baseline

# ==============================================================================
# PART 3: RANDOM VARIABLE CLASSIFICATION
# ==============================================================================

# Classification dictionary mapping variables to their statistical properties
classifications = {
    'days_active': {
        'type': 'discrete',
        'support': 'non-negative integers {0, 1, 2, ...}',
        'distribution': 'Poisson',
        'justification': 'Represents a count of whole days an account has been active.'
    },
    'score_views': {
        'type': 'discrete',
        'support': 'non-negative integers {0, 1, 2, ...}',
        'distribution': 'Poisson',
        'justification': 'Count of independent events (score views) over a fixed period.'
    },
    'session_minutes': {
        'type': 'continuous',
        'support': 'Non-negative real numbers [0, ∞)',
        'distribution': 'Log-Normal',
        'justification': 'Time-based measurement that can take any real value and is typically right-skewed.'
    },
    'risk_profile': {
        'type': 'categorical',
        'support': "{'conservative', 'balanced', 'aggressive'}",
        'distribution': 'Multinomial',
        'justification': 'Qualitative categories representing distinct, non-numeric risk levels.'
    },
    'premium_user': {
        'type': 'binary',
        'support': '{0, 1}',
        'distribution': 'Bernoulli',
        'justification': 'A single trial with exactly two outcomes (Success/Failure).'
    }
}

# ==============================================================================
# VALIDATION CHECKS (Autograder Safety)
# ==============================================================================

assert 0 <= p_premium <= 1, "P(premium) must be between 0 and 1"
assert 0 <= p_high_engagement <= 1, "P(high engagement) must be between 0 and 1"
assert 0 <= p_aggressive <= 1, "P(aggressive) must be between 0 and 1"
assert 0 <= p_joint <= 1, "Joint probability must be between 0 and 1"
assert abs(p_premium_given_engaged - bayes_check) < 0.01, "Bayes' theorem verification failed"
assert odds_ratio > 0, "Odds ratio must be positive"
assert all(classifications[var]['type'] for var in classifications), "Incomplete classifications"

# ==============================================================================
# BUSINESS OUTPUT & INTERPRETATION
# ==============================================================================

print("\n" + "="*80)
print("MILESTONE 1: PROBABILITY FOUNDATIONS")
print("="*80)

print("\n[PART 1: BASIC PROBABILITIES]")
print(f"P(premium user): {p_premium:.1%}")
print(f"  -> Interpretation: This is our baseline conversion rate for the entire dataset.")
print(f"P(high engagement): {p_high_engagement:.1%}")
print(f"  -> Interpretation: Percentage of users viewing scores 5 or more times.")
print(f"P(aggressive risk profile): {p_aggressive:.1%}")
print(f"  -> Interpretation: Proportion of the user base identifying as aggressive investors.")
print(f"Joint probability (Engaged AND Premium): {p_joint:.1%}")
print(f"  -> Interpretation: Probability of a user being both active and a paying customer.")

print("\n" + "-"*80)
print("[PART 2: CONDITIONAL PROBABILITY & BAYES]")
print(f"P(premium | engaged): {p_premium_given_engaged:.1%}")
print(f"P(engaged | premium): {p_engaged_given_premium:.1%}")
print(f"Bayes' Verification:  {bayes_check:.3f} = {p_premium_given_engaged:.3f} (Success)")
print(f"Odds Ratio:           {odds_ratio:.2f}x")
print(f"  -> Recommendation: Active engagement (3+ views) increases premium odds by {odds_ratio:.2f}x. "
      "Gamifying score views should be a priority.")

print("\n" + "-"*80)
print("[PART 3: VARIABLE CLASSIFICATION]")
print(f"{'Variable':<20} {'Type':<12} {'Distribution':<14} {'Support'}")
print("-" * 80)
for var, props in classifications.items():
    print(f"{var:<20} {props['type']:<12} {props['distribution']:<14} {props['support']}")

print("\n" + "="*80)
print("CRITICAL THINKING")
print("Question: Why is score_views discrete but session_minutes continuous?")
print("Answer:   score_views represents a count of distinct events that cannot be fractional, "
      "\n          whereas session_minutes is a measurement of time, which can be infinitely "
      "\n          divided into smaller decimal units.")
print("="*80 + "\n")
