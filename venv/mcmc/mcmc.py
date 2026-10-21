import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Define the Target Distribution (The Distribution We Want to Sample From) ---
# We use a simple Gaussian (Normal) distribution for this example.
# MCMC is most useful when this distribution is complex or unknown.

TARGET_MEAN = 10.0
TARGET_STD = 2.0

def target_distribution_pdf(x):
    """
    The Probability Density Function (PDF) of our target distribution.
    We are trying to sample points (x) that follow this distribution.
    """
    return norm.pdf(x, loc=TARGET_MEAN, scale=TARGET_STD)

# --- 2. Implement the Metropolis-Hastings Algorithm ---

def run_metropolis_hastings(n_steps, initial_state, proposal_std):
    """
    Performs the Metropolis-Hastings MCMC simulation.

    Args:
        n_steps (int): The number of steps (samples) to take.
        initial_state (float): The starting point of the chain.
        proposal_std (float): The standard deviation for the proposal function (the jump size).
    
    Returns:
        numpy.ndarray: The sequence of accepted samples.
    """
    # Initialize the chain
    chain = [initial_state]
    current_state = initial_state
    
    # Loop through the steps
    for _ in range(n_steps):
        # 1. Proposal: Propose a new state from a symmetric proposal distribution 
        # (a simple Gaussian centered at the current state is common).
        # We use a standard deviation (`proposal_std`) to control the size of the jump.
        proposed_state = np.random.normal(current_state, proposal_std)
        
        # 2. Acceptance Probability: Calculate the acceptance ratio (A)
        # Since the proposal is symmetric (q(x|y) = q(y|x)), the ratio simplifies:
        # A = min(1, [Target_PDF(Proposed) / Target_PDF(Current)])
        
        prob_proposed = target_distribution_pdf(proposed_state)
        prob_current = target_distribution_pdf(current_state)
        
        # Handle division by zero for distributions with zero probability
        if prob_current == 0:
            acceptance_ratio = 1.0 # Always accept if current state has 0 probability
        else:
            acceptance_ratio = min(1.0, prob_proposed / prob_current)
        
        # 3. Acceptance Step: Decide whether to move to the proposed state
        if np.random.rand() < acceptance_ratio:
            # Move accepted!
            current_state = proposed_state
        # else: Move rejected! current_state remains the same

        # Add the current state (accepted or rejected) to the chain
        chain.append(current_state)
        
    return np.array(chain)

# --- 3. Run the Simulation and Plot Results ---

# MCMC parameters
N_STEPS = 5000
BURN_IN = 500  # Discard initial samples (the "burn-in" period)
INITIAL_STATE = 0.0 # Start far from the mean
PROPOSAL_STD = 1.0 # Standard deviation for the proposal jump

# Run the simulation
mcmc_samples = run_metropolis_hastings(N_STEPS, INITIAL_STATE, PROPOSAL_STD)

# Apply burn-in
final_samples = mcmc_samples[BURN_IN:]

final_samples.sort()

# --- Plotting the Results ---
plt.figure(figsize=(12, 5))

# Plot 1: Histogram of the MCMC Samples
plt.subplot(1, 2, 1)
# Plot the histogram of the MCMC samples
plt.hist(final_samples, bins=20, density=True, alpha=0.6, color='skyblue', label='MCMC Samples')

# Plot the True Target Distribution PDF over the histogram
x_range = np.linspace(TARGET_MEAN - 4 * TARGET_STD, TARGET_MEAN + 4 * TARGET_STD, 1000)
plt.plot(x_range, target_distribution_pdf(x_range), 'r--', linewidth=2, label='True Target Distribution')

plt.title('Distribution of MCMC Samples')
plt.xlabel('Value (x)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(axis='y', alpha=0.5)

# Plot 2: Trace Plot (The Markov Chain)
plt.subplot(1, 2, 2)
plt.plot(mcmc_samples, color='gray', alpha=0.6, linewidth=0.5, label='Chain Trace')
plt.axvline(x=BURN_IN, color='red', linestyle='--', label=f'Burn-in end ({BURN_IN} steps)')
plt.title('MCMC Chain Trace')
plt.xlabel('Step Number')
plt.ylabel('Sampled Value')
plt.legend()
plt.grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.show()