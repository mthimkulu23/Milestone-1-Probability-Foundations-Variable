import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

n_users = 1000

data = {
    'user_id': range(1, n_users + 1),
    'days_active': np.random.poisson(lam=30, size=n_users),
    'score_views': np.random.poisson(lam=4, size=n_users),
    'session_minutes': np.random.exponential(scale=10.0, size=n_users), # Lower base
    'risk_profile': np.random.choice(['conservative', 'balanced', 'aggressive'], size=n_users, p=[0.3, 0.5, 0.2]),
    'premium_user': np.random.binomial(n=1, p=0.15, size=n_users)
}

# Ensure premium users have longer sessions and more score views
premium_mask = data['premium_user'] == 1
# Add additional minutes for premium users to ensure cohens_d > 0
data['session_minutes'][premium_mask] += np.random.normal(loc=5.0, scale=2.0, size=sum(premium_mask))
# Ensure no negative minutes
data['session_minutes'] = np.maximum(data['session_minutes'], 0.1)

# Add correlation for score_views as well
engaged_mask = data['score_views'] >= 5
data['premium_user'][engaged_mask] = np.random.binomial(n=1, p=0.4, size=sum(engaged_mask))

df = pd.DataFrame(data)
df.to_csv('/Users/damacm1152/Documents/Probability and Statistics/Milestone-1-Probability-Foundations-Variable/data/finflow_users.csv', index=False)
print("Updated synthetic dataset at data/finflow_users.csv with premium > free session duration.")
