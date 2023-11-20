from pathlib import Path
from allensdk.core.cell_types_cache import CellTypesCache
import numpy as np
import nemos as nmo
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV
import jax.numpy as jnp
import jax
from sklearn.model_selection import KFold
from sklearn.linear_model import PoissonRegressor
from utils import load_data
from utils import process_data
import itertools
import pynapple as nap

recording_id = 609492577
(
    time_trials,
    stim_trials,
    volt_trials,
    spike_counts,
    sweap_metadata,
    spike_times,
) = load_data.load_from_sdk(recording_id)

select_square = [
    key for key, value in sweap_metadata.items() if value == b"Long Square"
]
select_noise = [key for key, value in sweap_metadata.items() if value == b"Noise 2"]

# normalize stim to noise (noise assumes higher vals)
norm_stim_train = [
    stim_trials[tr].reshape(-1, 1) / stim_trials[select_noise[0]].max()
    for tr in select_square
]
counts_train = [spike_counts[tr].reshape(-1, 1) for tr in select_square]

# window size in ms
dt_ms = 0.0005
ws_ms_stim = int(200 / (1000 * dt_ms))
ws_ms_acg = int(150 / (1000 * dt_ms))
n_basis_acg = 10
n_basis_stim = 10


proc = process_data.TrialHandlerAllen(
    id_recording=recording_id, scale=stim_trials[select_noise[0]].max()
)

# proc.set_basis_stim(nsl.basis.BSplineBasis, ws_ms_stim, dict(n_basis_funcs=n_basis_stim, order=4))
proc.set_basis_stim(
    nmo.basis.RaisedCosineBasisLog, ws_ms_stim, dict(n_basis_funcs=n_basis_stim)
)

proc.set_basis_acg(
    nmo.basis.RaisedCosineBasisLog, ws_ms_acg, dict(n_basis_funcs=n_basis_acg)
)
# proc.set_basis_acg(nsl.basis.BSplineBasis, ws_ms_acg, dict(n_basis_funcs=n_basis_acg, order=2))
X, y, time = proc.get_convolved_predictor("Long Square", "Short Square")

alpha_grid = np.logspace(-6, -2, 10)
# model_skl = PoissonRegressor(max_iter=10**3, fit_intercept=True)
# kfold = KFold(n_splits=5, shuffle=True, random_state=123)
# cls = GridSearchCV(model_skl, n_jobs=-1, cv=kfold, param_grid={"alpha": alpha_grid})
# cls.fit(X[:, 0], y[:, 0])
# model_skl = cls.best_estimator_
# # model definition
# solver_kwargs = {
#     'tol': 10**-8,
#     'maxiter': 1000,
#     'jit': True
# }
#
# solver = nmo.solver.RidgeSolver('LBFGS', solver_kwargs=solver_kwargs, regularizer_strength=alpha_grid[1])
# obs_model = nmo.observation_models.PoissonObservations(inverse_link_function=jax.numpy.exp)
# model = nmo.glm.GLMRecurrent(solver=solver, observation_model=obs_model)
# model.coef_ = jnp.asarray(model_skl.coef_.reshape(1, -1))
# model.intercept_ = jnp.asarray(model_skl.intercept_.reshape(1,))
# y_sim, rate_sim = model.simulate_recurrent(
#         random_key=jax.random.PRNGKey(123),
#         feedforward_input=X[..., n_basis_acg:],
#         coupling_basis_matrix=proc.eval_basis_acg,
#         init_y=jnp.zeros((ws_ms_acg, 1))
#     )
# print(model.coef_)
# plt.plot(rate_sim)
#
# plt.plot(y_sim)


# loop over number of basis, time points and regularizers
grid_param = itertools.product(alpha_grid, range(7, 15), range(150, 400, 50))
result_params = {}
for alpha, n_basis_acg, window_size in grid_param:
    print(alpha, n_basis_acg, window_size)
    solver = nmo.solver.RidgeSolver(
        "GradientDescent", solver_kwargs={"jit": True}, regularizer_strength=alpha
    )
    obs_model = nmo.observation_models.PoissonObservations(
        inverse_link_function=jax.nn.softplus
    )
    model = nmo.glm.GLMRecurrent(solver=solver, observation_model=obs_model)
    ws_ms_stim = int(200 / (1000 * dt_ms))
    ws_ms_acg = int(window_size / (1000 * dt_ms))
    proc.set_basis_stim(
        nmo.basis.RaisedCosineBasisLog, ws_ms_stim, dict(n_basis_funcs=n_basis_stim)
    )
    proc.set_basis_acg(
        nmo.basis.RaisedCosineBasisLog, ws_ms_acg, dict(n_basis_funcs=n_basis_acg + 1)
    )
    proc.eval_basis_acg = proc.eval_basis_acg[:, 1:]
    X, y = proc.get_convolved_predictor("Long Square", "Short Square")
    init_params = np.zeros((1, n_basis_stim + n_basis_acg)), np.log(
        np.array([y.mean()])
    )
    model.fit(X, y, init_params=init_params)
    result_params[(alpha, n_basis_acg, window_size)] = {
        "coef_": model.coef_,
        "intercept_": model.intercept_,
        "basis_acg": proc.eval_basis_acg,
        "basis_stim": proc.eval_basis_stim,
    }

# transform to numpy and save
for key in result_params.keys():
    result_params[key]["coef_"] = np.asarray(result_params[key]["coef_"])
    result_params[key]["intercept_"] = np.asarray(result_params[key]["intercept_"])
np.savez(
    "grid_fit_all_basis.npz",
    obs_noise="nmo.observation_models.PoissonObservations",
    link_func="jax.nn.softplus",
    results=result_params,
    basis_type="nmo.basis.RaisedCosineBasisLog",
)


# syntehtic data
#
# y_sint = np.tile(y.flatten()[2*282600:(2*283000-10)],1000)
# X_sint = np.hstack(
#     [np.convolve(y_sint, proc.eval_basis_acg[:, k], mode='valid').reshape(-1, 1) for k in range(n_basis_acg)]
# )
# model_skl = PoissonRegressor(alpha=alpha, max_iter=10**3, fit_intercept=True)
# model_skl.fit(X_sint[:-1], y_sint[ws_ms_acg:])
