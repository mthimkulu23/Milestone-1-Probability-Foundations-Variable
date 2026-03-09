import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import os

# Load dataset
file_path = os.path.join('data', 'finflow_users.csv')
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"❌ Error: Dataset not found at {file_path}. Please ensure the 'data' folder contains 'finflow_users.csv'.")
    exit()

# ============================================
# PART 1: SAMPLE SPACES & BASIC PROBABILITY
# ============================================

# TODO: Define sample space for premium_user
sample_space_premium = set(df['premium_user'].unique())

# TODO: Calculate P(premium_user = 1)
p_premium = (df['premium_user'] == 1).mean()

# TODO: Calculate P(score_views >= 5)
p_high_engagement = (df['score_views'] >= 5).mean()

# TODO: Calculate P(risk_profile = 'aggressive')
p_aggressive = (df['risk_profile'] == 'aggressive').mean()

# TODO: Calculate joint probability P(score_views >= 5 AND premium_user = 1)
p_joint = ((df['score_views'] >= 5) & (df['premium_user'] == 1)).mean()

# ============================================
# PART 2: CONDITIONAL PROBABILITY & BAYES
# ============================================

# TODO: Calculate P(premium = 1 | score_views >= 3)
engaged_mask = df['score_views'] >= 3
p_premium_given_engaged = df[engaged_mask]['premium_user'].mean()

# TODO: Calculate P(score_views >= 3 | premium = 1)
premium_mask = df['premium_user'] == 1
p_engaged_given_premium = df[premium_mask]['score_views'].ge(3).mean()

# TODO: Calculate P(score_views >= 3)
p_engaged = engaged_mask.mean()

# TODO: Verify Bayes' theorem
# bayes_check should equal p_premium_given_engaged within 0.01 tolerance
bayes_check = (p_engaged_given_premium * p_premium) / p_engaged

# TODO: Calculate odds ratio
# odds_ratio = [P(premium|engaged) / (1 - P(premium|engaged))] / [P(premium) / (1 - P(premium))]
odds_engaged = p_premium_given_engaged / (1 - p_premium_given_engaged)
odds_baseline = p_premium / (1 - p_premium)
odds_ratio = odds_engaged / odds_baseline

# ============================================
# PART 3: RANDOM VARIABLE CLASSIFICATION
# ============================================

# TODO: Create classification dictionary
classifications = {
    'days_active': {
        'type': 'discrete',
        'support': 'non-negative integers {0, 1, 2, ...}',
        'distribution': 'Poisson',
        'justification': 'The number of days an account has been active is a count of discrete units of time.'
    },
    'score_views': {
        'type': 'discrete',
        'support': 'non-negative integers {0, 1, 2, ...}',
        'distribution': 'Poisson',
        'justification': 'Count of independent events (score views) over fixed period.'
    },
    'session_minutes': {
        'type': 'continuous',
        'support': 'Non-negative real numbers [0, ∞)',
        'distribution': 'Log-Normal',
        'justification': 'Session duration can be any real value and is typically right-skewed with most sessions being short.'
    },
    'risk_profile': {
        'type': 'categorical',
        'support': "{'conservative', 'balanced', 'aggressive'}",
        'distribution': 'Multinomial',
        'justification': 'Users are grouped into distinct, non-numeric categories that represent different risk levels.'
    },
    'premium_user': {
        'type': 'binary',
        'support': '{0, 1}',
        'distribution': 'Bernoulli',
        'justification': 'A single trial with two possible outcomes: converted to premium (1) or not (0).'
    }
}

# Validation checks (do not modify)
assert 0 <= p_premium <= 1, "P(premium) must be between 0 and 1"
assert 0 <= p_high_engagement <= 1, "P(high engagement) must be between 0 and 1"
assert 0 <= p_aggressive <= 1, "P(aggressive) must be between 0 and 1"
assert 0 <= p_joint <= 1, "Joint probability must be between 0 and 1"
assert abs(p_premium_given_engaged - bayes_check) < 0.01, "Bayes' theorem verification failed"
assert odds_ratio > 0, "Odds ratio must be positive"
assert all(classifications[var]['type'] for var in classifications), "All variables must be classified"

# ============================================
# BUSINESS INTERPRETATION
# ============================================

print("="*70)
print("PART 1: BASIC PROBABILITIES")
print("="*70)
print(f"P(premium user): {p_premium:.1%}")
print(f"  → Interpretation: This represents the baseline conversion rate of our current user base.")
print(f"\nP(high engagement): {p_high_engagement:.1%}")
print(f"  → Interpretation: This represents the proportion of users who are heavily interacting with the scoring feature.")
print(f"\nP(aggressive risk profile): {p_aggressive:.1%}")
print(f"  → Interpretation: This tells us how many users identify with a high-risk investment strategy.")
print(f"\nJoint probability (high engagement AND premium): {p_joint:.1%}")
print(f"  → Interpretation: This shows the core overlap between our most active and paying segments.")

print("\n" + "="*70)
print("PART 2: CONDITIONAL PROBABILITY & BAYES")
print("="*70)
print(f"P(premium | engaged): {p_premium_given_engaged:.1%}")
print(f"P(engaged | premium): {p_engaged_given_premium:.1%}")
print(f"\nBayes' theorem verification: {bayes_check:.3f} ≈ {p_premium_given_engaged:.3f} ✓")
print(f"\nOdds ratio: {odds_ratio:.2f}x")
print(f"  → Interpretation: Engagement (score_views >= 3) increases the odds of premium conversion by {odds_ratio:.2f}x. We should focus on driving users to view their scores to improve conversion.")

print("\n" + "="*70)
print("PART 3: VARIABLE CLASSIFICATION")
print("="*70)
print(f"{'Variable':<20} {'Type':<15} {'Distribution':<15} {'Support'}")
print("-"*70)
for var, props in classifications.items():
    print(f"{var:<20} {props['type']:<15} {props['distribution']:<15} {props['support']}")
print("="*70)

# Critical thinking question
print("\n❓ Why is score_views discrete but session_minutes continuous?")
print("   → Answer: score_views is discrete because it represents a countable number of individual events (you can't have 2.5 views), while session_minutes is continuous because it measures time, which can take any real value within an interval (e.g., 2.53 minutes) depending on the precision of measurement.")
