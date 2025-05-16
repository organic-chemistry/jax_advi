# JAX ADVI

#### What is this?

JAX ADVI is a small library designed to do black-box variational inference using automatic differentiation variational inference ([ADVI](https://jmlr.org/papers/v18/16-107.html)) using [JAX](https://github.com/google/jax). Specifically, it implements the variant proposed by [Giordano et al.](https://jmlr.org/papers/v19/17-670.html) which allows the use of (approximate) second order methods, which can be more robust. You can read about the methodology [on my blog](https://martiningram.github.io/deterministic-advi/).

#### How do I use it?

##### Installation

To use this library, you'll need [JAX](https://github.com/google/jax). If you want to run the examples, you'll also need:

* Pandas
* Scikit-learn
* Matplotlib
* Pystan

To install, clone this repository and run `python setup.py develop`.

##### Guide Types

JAX ADVI supports several types of variational guides to approximate the posterior distribution. The primary ones you'll interact with are:

*   **`MAPGuide`**: Performs Maximum A Posteriori (MAP) estimation. This finds the mode of the posterior distribution, which can be a good point estimate or a starting point for more complex variational inference.
*   **`MeanFieldGuide`**: Assumes a factorized (mean-field) Gaussian approximation to the posterior. This is a common and often effective choice for ADVI.
*   **`LowRankGuide`**: Extends the `MeanFieldGuide` by adding a low-rank correction to the covariance matrix, allowing for some correlations between parameters.
*   **`FullRankGuide`**: Assumes a full-rank Gaussian approximation, capturing all correlations between parameters. This is the most flexible but also the most computationally intensive.

With these guides, you can:

*   Perform MAP estimation.
*   Draw samples from the (approximate) posterior distribution.
*   Obtain the mean and standard deviation of the approximate posterior.

**Initializing `MeanFieldGuide` from `MAPGuide`**

It can be beneficial to first find the MAP estimate and then use it to initialize the parameters of a `MeanFieldGuide`. Here's how you can do this using the logistic regression example:

First, perform MAP estimation:
```python
from jax_advi.advi import optimize_advi, ADVI_params # Assuming ADVI_params is here
from jax_advi.guide import MAPGuide, MeanFieldGuide
# ... (rest of the logistic regression setup: theta_shapes, calculate_likelihood, calculate_prior, log_lik_fun, log_prior_fun)

# Initial MAP fit
map_guide = MAPGuide()
map_result = optimize_advi(
    theta_shapes,
    log_prior_fun,
    log_lik_fun,
    guide=map_guide,
    # optimizer_kwargs, num_steps etc. might be needed depending on optimize_advi signature
)

# Extract MAP estimates to initialize MeanFieldGuide
# The exact way to get `init_mean_scale` depends on the structure of `map_result`
# Assuming map_result['params'] holds the flat optimized parameters
init_mean_scale = map_result['params'] 
# init_log_sd_scale can be a small negative number for initial uncertainty
init_log_sd_scale = -1.0 

# Then, fit with MeanFieldGuide
mean_field_guide = MeanFieldGuide()
mean_field_guide.init_mean_scale = init_mean_scale
mean_field_guide.init_log_sd_scale = init_log_sd_scale

result = optimize_advi(
    theta_shapes, 
    log_prior_fun, 
    log_lik_fun, 
    guide=mean_field_guide # Pass the initialized guide
    # n_draws=None # or your desired number of draws
)
```

##### Usage

The key function is `optimize_advi`. Here's a simple example of a logistic regression:

Model:

![log_reg_model](images/log_reg_model.png)

Code:
```python
from jax_advi.advi import optimize_advi
from jax import jit
from jax.scipy.stats import norm
from functools import partial
from jax.nn import log_sigmoid

# Define parameter shapes:
theta_shapes = {
    'beta': (K),
    'gamma': ()
}

# Define a function to calculate the log likelihood
def calculate_likelihood(theta, X, y):
    
    logit_prob = X @ theta['beta'] + theta['gamma']
    
    prob_pres = log_sigmoid(logit_prob)
    prob_abs = log_sigmoid(-logit_prob)
    
    return jnp.sum(y * prob_pres + (1 - y) * prob_abs)

# Define a function to calculate the log prior
def calculate_prior(theta):
    
    beta_prior = jnp.sum(norm.logpdf(theta['beta']))
    gamma_prior = jnp.sum(norm.logpdf(theta['gamma']))
    
    return beta_prior + gamma_prior
	
# The partial application basically conditions on the data (not defined in this
# little snippet)
log_lik_fun = jit(partial(calculate_likelihood, X=X, y=y))
log_prior_fun = jit(calculate_prior)

# Call the optimisation function
result = optimize_advi_mean_field(theta_shapes, log_prior_fun, log_lik_fun, n_draws=None)
```

JAX ADVI should typically get the means right, but the variances may be off. For a full example comparing against Stan's HMC, see [the example notebook](https://github.com/martiningram/jax_advi/blob/main/examples/Logistic%20regression%20example.ipynb). Here is a comparison for some toy data, taken from that example:

![comparison](images/logistic_reg_comparison.png)
As promised, good means, less reliable standard deviations.

##### Example 2

A more complex example is the following hierarchical model:

![bradley_terry](images/bradley_terry.png)

You can read a bit more about it in [the blog post](https://martiningram.github.io/deterministic-advi/) if you like. Here's what the code looks like:

Define shapes:

```python
theta_shapes = {
    'player_skills': (n_p),
    'skill_prior_sd': ()
}
```

Constrain the prior standard deviation to be positive:

```python
from jax_advi.constraints import constrain_positive

theta_constraints = {
    'skill_prior_sd': constrain_positive
}
```

Define the log likelihood and prior:

```python
from jax.scipy.stats import norm
from jax import jit
from jax.nn import log_sigmoid
import jax.numpy as jnp

@jit
def log_prior_fun(theta):
    
    # Prior
    skill_prior = jnp.sum(norm.logpdf(theta['player_skills'], 0., theta['skill_prior_sd']))
    
    # hyperpriors
    hyper_sd = norm.logpdf(theta['skill_prior_sd'])
    
    return skill_prior + hyper_sd

def log_lik_fun(theta, winner_ids, loser_ids):
    
    logit_probs = theta['player_skills'][winner_ids] - theta['player_skills'][loser_ids]
    
    return jnp.sum(log_sigmoid(logit_probs))

from functools import partial

curried_lik = jit(partial(log_lik_fun, winner_ids=winner_ids, loser_ids=loser_ids))
```

Finally, optimize:

```python
result = optimize_advi_mean_field(
	theta_shapes, log_prior_fun, curried_lik, 
	constrain_fun_dict=theta_constraints, verbose=True, M=100)
```

You can run the example and compare it against Stan [using the example notebook](https://github.com/martiningram/jax_advi/blob/main/examples/Tennis%20example.ipynb). Again, the means are good and the variances are a bit off:

![tennis_comparison](images/tennis_comparison.png)
The fit took 10s on my laptop with a GTX 2070, compared to 100 minutes in Stan, so you do get quite a speedup. It's also more accurate than Stan's ADVI, which gives the following means:

![stan_advi](images/tennis_comparison_stan_advi.png)
I hope you find this library useful. Please raise issues if anything doesn't work. Please note that this is still a new package and I wouldn't trusting it blindly yet. Its mean estimates seem reliable so far, but I recommend checking them against Stan's (or another MCMC framework) to be sure. If you find examples that break it, I'd be very interested to see them.
