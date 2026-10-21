import numpy as np
import matplotlib.pyplot as plt

# --- 1. Data and Parameters from the Previous Example ---
TRUE_ALPHA = 1.5
TRUE_BETA = -0.7
SIGMA_NOISE = 1.0 
X = np.linspace(0, 10, 50)
Y = TRUE_ALPHA + TRUE_BETA * X + np.random.normal(0, SIGMA_NOISE, 50)

# --- 2. Metropolis-Hastings Chain (re-run for plotting) ---
# Parameters
PRIOR_MU = 0.0
PRIOR_TAU = 10.0
N_STEPS = 50000
BURN_IN = 5000
PROPOSAL_STDEV = 0.1 

def log_posterior(params, x, y, sigma_noise):
    alpha, beta = params
    mu_model = alpha + beta * x
    log_likelihood = -0.5 * np.sum((y - mu_model)**2 / sigma_noise**2)
    log_prior_alpha = -0.5 * ((alpha - PRIOR_MU) / PRIOR_TAU)**2
    log_prior_beta = -0.5 * ((beta - PRIOR_MU) / PRIOR_TAU)**2
    return log_likelihood + log_prior_alpha + log_prior_beta

current_params = np.array([0.0, 0.0]) 
chain = []

for i in range(N_STEPS):
    proposed_params = current_params + np.random.normal(0, PROPOSAL_STDEV, size=2)
    
    log_f_proposed = log_posterior(proposed_params, X, Y, SIGMA_NOISE)
    log_f_current = log_posterior(current_params, X, Y, SIGMA_NOISE)
    
    log_ratio = log_f_proposed - log_f_current
    log_alpha = min(0.0, log_ratio)
    
    if np.log(np.random.rand()) < log_alpha:
        current_params = proposed_params
        
    chain.append(current_params)

chain = np.array(chain)
final_samples = chain[BURN_IN:]

# Calculate Posterior Means
mean_alpha = np.mean(final_samples[:, 0])
mean_beta = np.mean(final_samples[:, 1])

# --- 3. Visualization Code ---

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Plot 1: MCMC Trace Plot (Alpha) ---
axes[0, 0].plot(chain[:1000, 0], label='Chain', color='C0', alpha=0.8)
axes[0, 0].axvline(BURN_IN, color='r', linestyle='--', label='Burn-in end')
axes[0, 0].set_title('MCMC Trace Plot for $\\alpha$ (Intercept)')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('$\\alpha$ value')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle=':', alpha=0.6)

# --- Plot 2: MCMC Trace Plot (Beta) ---
axes[0, 1].plot(chain[:1000, 1], label='Chain', color='C1', alpha=0.8)
axes[0, 1].axvline(BURN_IN, color='r', linestyle='--', label='Burn-in end')
axes[0, 1].set_title('MCMC Trace Plot for $\\beta$ (Slope)')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('$\\beta$ value')
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle=':', alpha=0.6)

# --- Plot 3: Joint Posterior Distribution (Scatter/2D Histogram) ---
axes[1, 0].hist2d(final_samples[:, 0], final_samples[:, 1], bins=30, cmap='viridis', density=True)
axes[1, 0].scatter(mean_alpha, mean_beta, color='r', marker='x', s=100, label='Posterior Mean')
axes[1, 0].scatter(TRUE_ALPHA, TRUE_BETA, color='w', marker='o', edgecolors='k', s=100, label='True Values')
axes[1, 0].set_title('Joint Posterior Density for $(\\alpha, \\beta)$')
axes[1, 0].set_xlabel('$\\alpha$')
axes[1, 0].set_ylabel('$\\beta$')
axes[1, 0].legend()

# --- Plot 4: Regression Lines (Posterior Predictive Check) ---
axes[1, 1].scatter(X, Y, label='Observed Data', color='k', s=20)
axes[1, 1].plot(X, TRUE_ALPHA + TRUE_BETA * X, color='r', linestyle='--', label='True Regression Line')

# Plot a few sampled regression lines
num_lines_to_plot = 100
for i in np.random.choice(len(final_samples), num_lines_to_plot, replace=False):
    alpha_s = final_samples[i, 0]
    beta_s = final_samples[i, 1]
    axes[1, 1].plot(X, alpha_s + beta_s * X, color='teal', alpha=0.05)
    
axes[1, 1].plot([], [], color='teal', alpha=0.5, label=f'{num_lines_to_plot} Sampled Lines') # Placeholder for legend
axes[1, 1].set_title('Posterior Predictive Lines')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('.\\venv\\mcmc\\mcmc_bayesian_regression_plots.png')