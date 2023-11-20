"""
## Virtual Environment and Allen SDK

The system Python on macOS 12.6.5 is Python 3.9.6 whose ssl module is
compiled with LibreSSL 2.8.3.

If your macOS default python relies on LibreSSL, your virtual python environment created
with virtualenv will rely on LibreSSL too. this will be incompatible with the allensdk which depends on urllib3,
that requires OpenSSL instead.

To check your ssl, run the following python commands.

```python

import ssl

# provides a string representation of the OpenSSL (or LibreSSL) version.
print(ssl.OPENSSL_VERSION)

```
If you have the OpenSSL version installed, you are good to go.
If you have the LibreSSL, one simple way out of this is to create your allensdk environment using conda.

### Installing jax in a conda environment

If your allensdk is installed in a conda environment, it is recommended to
install `jax` and `jaxlib` from conda-forge instead of pip.

```python
# If you have already installed jax and jaxlib:
pip uninstall jax jaxlib

# install packages from conda-forge
conda install -c conda-forge jaxlib
conda install -c conda-forge jax
```

"""
from pathlib import Path
from allensdk.core.cell_types_cache import CellTypesCache
import numpy as np
import nemos as nmo
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV
import jax.numpy as jnp
import jax
from sklearn.model_selection import KFold


def scaled_coincidence_rate(
        true_spk_ts:np.ndarray,
        model_spk_ts: np.ndarray,
        recording_duration_ms: float,
        resolution_ms: float = 4.
):
    """
    Measure of spike time precision from [1].

    Parameters
    ----------
    true_spk_ts
    model_spk_ts
    recording_duration_ms
    resolution_ms:

    Returns
    -------
    :
        The scaled coincidence rate.
    """
    # very memory inefficient, the simplest way possible
    coincident_events = np.sum(np.abs(true_spk_ts[:, None] - model_spk_ts) <= resolution_ms)
    n_true_spk = true_spk_ts.shape[0]
    n_sim_spk = model_spk_ts.shape[0]
    poisson_expected_events = 2 * resolution_ms * n_true_spk * n_sim_spk / recording_duration_ms

    scaled_coincicence = (
            (coincident_events - poisson_expected_events) /
            (0.5 * (1 - poisson_expected_events / n_true_spk) * (n_true_spk + n_sim_spk))
    )
    return scaled_coincicence

def plot_cv_results(cls):
    # Extract the results from the fitted GridSearchCV object
    cv_results = cls.cv_results_

    # Now, we'll extract the data for plotting
    means = cv_results['mean_test_score']
    params = cv_results['params']

    # Organize scores and alphas in nested dictionaries based on solver type and inverse link function
    scores_dict = {}
    alphas_dict = {}

    for mean, param in zip(means, params):
        # get the name of the parameters
        solver_type = type(param['solver']).__name__
        link_func = param['noise_model__inverse_link_function'].__name__

        if solver_type not in scores_dict:
            scores_dict[solver_type] = {}
            alphas_dict[solver_type] = {}

        if link_func not in scores_dict[solver_type]:
            scores_dict[solver_type][link_func] = []
            alphas_dict[solver_type][link_func] = []

        scores_dict[solver_type][link_func].append(mean)
        alphas_dict[solver_type][link_func].append(param['solver__regularizer_strength'])

    # Now, create the plot
    plt.figure(figsize=(10, 6))

    for solver_type, link_funcs_dict in scores_dict.items():
        for link_func, scores in link_funcs_dict.items():
            alphas = alphas_dict[solver_type][link_func]
            label = f"{solver_type}, {link_func}"
            plt.plot(alphas, scores, label=label, marker='o')

    plt.xscale('log')  # if you want a logarithmic x-axis
    plt.xlabel('Regularizer Strength')
    plt.ylabel('Mean Test Score')
    plt.legend(title="Configuration")
    plt.title('Mean Test Score as a Function of Regularizer Strength')
    plt.show()





data_path = Path("//cell_types")

# Initialize a cache for the cell types database
ctc = CellTypesCache()

# The id you've posted seems to be a specific recording (ephys) id, so you'd use:
data_set = ctc.get_ephys_data(609492577)

# print len trials
sweap_nums = data_set.get_sweep_numbers()
# keep only trials of 8sec dur
n_trials = len(sweap_nums)
sweap_nums = np.array(sweap_nums)

# dt for counting spikes

dt_sec = 0.001
stim_trials = {}
volt_trials = {}
spike_counts = {}
sweap_metadata = {}
time_trials = {}

# stack stim and resp in a matrix
cc = 0
for num in sweap_nums:

    # get the data for a specific trial
    dat = data_set.get_sweep(num)

    sweap_metadata[num] = data_set.get_sweep_metadata(num)['aibs_stimulus_name']

    # get the time for each sample
    time_samp = np.arange(dat['stimulus'].shape[0]) / dat['sampling_rate']

    # binning for spike times
    init_time = time_samp[0]
    end_time = time_samp[-1]

    time_sec = np.arange(0, int((end_time-init_time) / dt_sec) + 1) * dt_sec
    edge_spike_bin = np.hstack((time_sec, time_sec[-1] + dt_sec))

    stim_trials[num] = np.interp(time_sec, time_samp, dat['stimulus'])
    volt_trials[num] = np.interp(time_sec, time_samp, dat['response'])
    spike_counts[num] = np.histogram(data_set.get_spike_times(num), bins=edge_spike_bin)[0]
    time_trials[num] = time_sec
    cc += 1

# %%
# prepare model for stim
# select all noise stim
select_noise = [key for key, value in sweap_metadata.items() if value == b"Noise 2"]

norm_stim_train = stim_trials[select_noise[0]] / stim_trials[select_noise[0]].max()
counts_train = spike_counts[select_noise[0]]
counts_test = spike_counts[select_noise[1]]

# use the same normalizaiton
norm_stim_test = stim_trials[select_noise[1]] / stim_trials[select_noise[0]].max()


# window size in ms
ws_ms = 200
n_basis = 10
B = nmo.basis.RaisedCosineBasisLog(n_basis, alpha=1.).evaluate(np.linspace(0, 1, ws_ms))
conv_counts = nmo.utils.convolve_1d_trials(B, [counts_train.reshape(-1, 1)])
conv_stim = nmo.utils.convolve_1d_trials(B, [norm_stim_train.reshape(-1, 1)])

conv_counts_test = nmo.utils.convolve_1d_trials(B, [counts_test.reshape(-1, 1)])
conv_stim_test = nmo.utils.convolve_1d_trials(B, [norm_stim_test.reshape(-1, 1)])



# model

kfold = KFold(n_splits=5, shuffle=True, random_state=123)
jax.config.update("jax_enable_x64", True)


solver_kwargs = {
    'tol': 10**-8,
    'maxiter': 5000,
    'jit': True
}

grid_regul = np.logspace(-5, -2, 8)
parameter_grid_ridge = {
    'solver': [nmo.solver.RidgeSolver('LBFGS', solver_kwargs=solver_kwargs)],
    'solver__regularizer_strength': grid_regul,
    'noise_model__inverse_link_function': [jax.nn.softplus]
}

model = nmo.glm.GLMRecurrent()

y = jnp.asarray(counts_train[ws_ms:].reshape(-1, 1), dtype=jnp.float64)
X = jnp.asarray(np.concatenate(conv_counts + conv_stim, axis=2)[:-1], dtype=jnp.float64)


cls = GridSearchCV(model, param_grid=parameter_grid_ridge, cv=kfold,  n_jobs=-1)
cls.fit(X, y)

y_test = jnp.asarray(counts_test[ws_ms:].reshape(-1, 1), dtype=jnp.float64)
X_test = jnp.asarray(np.concatenate(conv_counts_test + conv_stim_test, axis=2)[:-1], dtype=jnp.float64)
best_score = cls.best_estimator_.score(X_test, y_test)
print(f'Best Score: {best_score}')

y_sim, rate_sim = cls.best_estimator_.simulate_recurrent(
    random_key=jax.random.PRNGKey(123),
    feedforward_input=X_test[..., n_basis:],
    coupling_basis_matrix=B,
    init_y=jnp.zeros((ws_ms, 1))
)

time_test = time_trials[select_noise[1]]
plt.figure()
plt.suptitle("Recurrent simulations - Test set")
plt.title(f"Regularizer strength: {cls.best_params_['solver__regularizer_strength']:.2e}")
plt.vlines(time_test[np.where(y_sim.flatten())[0]],0,1, label="simulated")
plt.vlines(time_test[np.where(counts_test.flatten()[ws_ms:])[0]],1.1,2.1,color='orange', label="recorded")
plt.plot(time_test, norm_stim_test-1.2, color='k', label="stimulus")
plt.xlabel('time[sec]')
plt.legend()


model_spk_time_ms = time_test[np.where(y_sim.flatten())[0]] * 1000.
true_spk_time_ms = time_test[np.where(counts_test.flatten()[ws_ms:])[0]] * 1000.
total_time = time_test[-1] * 1000.
scaled_coincidence = scaled_coincidence_rate(
    true_spk_time_ms,
    model_spk_time_ms,
    total_time,
    resolution_ms=4.
)


import sklearn.linear_model as lin

# fit results and simulation accuracy for different alpha
param_dict = {}
param_dict_skl = {}
pr2_dict = {}
pr2_dict_train = {}
pr2_train_skl = {}
scaled_coincidence_dict = {}
alpha_grid = np.logspace(-6, -2, 10)
for alpha in alpha_grid:
    print(f"alpha: {alpha:.2e}")
    model = nmo.glm.GLMRecurrent(solver=nmo.solver.RidgeSolver(solver_name="BFGS"))
    model.solver.solver_kwargs = {"tol": 10 ** -13, "jit": True}
    model.solver.regularizer_strength = alpha
    model.fit(X, y)
    y_sim_alpha, rate_sim_alpha = model.simulate_recurrent(
        random_key=jax.random.PRNGKey(123),
        feedforward_input=X_test[..., n_basis:],
        coupling_basis_matrix=B,
        init_y=jnp.zeros((ws_ms, 1))
    )

    # sklearn
    mdl = lin.PoissonRegressor(alpha=alpha, max_iter=10 ** 4)
    mdl.fit(X[:, 0], y[:, 0])
    param_dict_skl[alpha] = mdl.coef_
    param_dict[alpha] = model.basis_coeff_

    pr2_dict_train[alpha] = model.score(X, y, score_type="log-likelihood")
    #pr2_dict_train[alpha] = mdl.score(X[:, 0], y[:,0])


    #pr2_dict[alpha] = mdl.score(X_test[:,0], y_test[:,0])
    pr2_dict[alpha] = model.score(X_test, y_test, score_type="log-likelihood")

    model_spk_time_ms = time_test[np.where(y_sim_alpha.flatten())[0]] * 1000.
    true_spk_time_ms = time_test[np.where(counts_test.flatten()[ws_ms:])[0]] * 1000.
    total_time = time_test[-1] * 1000.

    scaled_coincidence_dict[alpha] = scaled_coincidence_rate(true_spk_time_ms, model_spk_time_ms, total_time,
                                                             resolution_ms=4.)

alphas = list(pr2_dict.keys())
pr2s = list(pr2_dict.values())
pr2s_train = list(pr2_dict_train.values())
scaled_coincidences = list(scaled_coincidence_dict.values())

# Create a 3x1 subplot layout
fig, ax = plt.subplots(1, 3, figsize=(12, 3))

# Plot PR^2
ax[0].semilogx(alphas, pr2s_train, '-o')
ax[0].set_title('PR^2 as a function of regularizer')
ax[0].set_xlabel('reg. strength')
ax[0].set_ylabel('train PR^2')

ax[1].semilogx(alphas, pr2s, '-o')
ax[1].set_title('PR^2 as a function of regularizer')
ax[1].set_xlabel('reg. strength')
ax[1].set_ylabel('test PR^2')

# Plot Scaled Coincidence
ax[2].semilogx(alphas, scaled_coincidences, '-o')
ax[2].set_title('Scaled Coincidence as a function of regularizer')
ax[2].set_xlabel('reg. strength')
ax[2].set_ylabel('Scaled Coincidence')

plt.tight_layout()
plt.show()


# def _clean(self, x):
#     """
#     Helper function to trim the data so that it is in (0,inf)
#
#     Notes
#     -----
#     The need for this function was discovered through usage and its
#     possible that other families might need a check for validity of the
#     domain.
#     """
#     return np.clip(x, FLOAT_EPS, np.inf)

# def _resid_dev(self, endog, mu):
#     endog_mu = self._clean(endog / mu)
#     resid_dev = endog * np.log(endog_mu) - (endog - mu)
#     return 2 * resid_dev