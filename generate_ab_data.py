import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

n_per_variant = 500
variants = ['control', 'variant_a', 'variant_b', 'variant_c', 'variant_d']

# Base conversion rate
base_rate = 0.12

# Treatment effects
effects = {
    'control': 0,
    'variant_a': 0.05,  # Strong significant effect
    'variant_b': 0.01,  # Small possibly non-significant effect
    'variant_c': -0.02, # Negative effect
    'variant_d': 0.04   # Moderate significant effect
}

data = []
for variant in variants:
    rate = base_rate + effects[variant]
    conversions = np.random.binomial(n=1, p=rate, size=n_per_variant)
    for conv in conversions:
        data.append({'variant': variant, 'converted': conv})

ab_df = pd.DataFrame(data)
ab_df.to_csv('/Users/damacm1152/Documents/Probability and Statistics/Milestone-1-Probability-Foundations-Variable/data/finflow_ab_test.csv', index=False)
print("Created synthetic A/B test dataset at data/finflow_ab_test.csv")
