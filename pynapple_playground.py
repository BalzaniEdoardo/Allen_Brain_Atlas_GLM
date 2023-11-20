import jax
import matplotlib.pylab as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap

from utils import load_data, process_data

# Load the recording information
info_recordings = pd.read_csv("cell_types_specimen_details.csv")
print(info_recordings.loc[
        2172,
        ['line_name', 'specimen__id', 'specimen__name', 'structure__id', 'structure__name', 'structure__acronym']
])


# select a recording
specimen_id = 609492577

# Parameters
train_trial_labels = "Long Square", "Short Square"
window_size_acg = 250
window_size_stim = 200
dt_sec = 0.0005
n_basis_acg = 10
n_basis_stim = 11

# load the experiment (creates pynapple objects)
experiment = process_data.PynappleLoader(specimen_id)

# select some stimulation protocols
experiment.set_trial_filter(*train_trial_labels)

# create nested dictionary with predictors, first key: "predictor name". second key: "trial ID"
experiment.create_predictors("spike_counts_0", "injected_current")

# add the predictors and the count to object that will
# construct the model design matrix
# the bin size is used to determine the basis window size
data_handler = process_data.ModelConstructor(
        predictor_dict=experiment.predictor_dict,
        counts_dict=experiment.spike_counts_dict,
        bin_size_sec=experiment.bin_size_sec
)
data_handler.set_basis(
        "spike_counts_0",
        nmo.basis.RaisedCosineBasisLog,
        window_size_acg,
        dict(n_basis_funcs=n_basis_acg)
)
data_handler.set_basis(
        "injected_current",
        nmo.basis.RaisedCosineBasisLog,
        window_size_stim,
        dict(n_basis_funcs=n_basis_stim)
)
X, y, valid = data_handler.get_convolved_predictor()


# Model definition
regularizer_strength = 8 * 10**-11
solver = nmo.solver.RidgeSolver(
    "GradientDescent",
    solver_kwargs={"jit": True},
    regularizer_strength=regularizer_strength,
)
obs_model = nmo.observation_models.PoissonObservations(
    inverse_link_function=jax.nn.softplus
)
model = nmo.glm.GLMRecurrent(solver=solver, observation_model=obs_model)

# initialize and fit the model
init_params = np.zeros((1, n_basis_stim + n_basis_acg)), np.log(np.mean(y, axis=0))
model.fit(X[valid], y[valid], init_params=init_params)
rate = model.predict(X[valid])

# convert back to pynapple
time = experiment.get_time()
rate_nap = nap.TsdFrame(
    t=time[valid],
    d=np.asarray(rate),
    time_support=experiment.spike_times.time_support
)
trig_average = nap.compute_event_trigger_average(
    experiment.spike_times,
    rate_nap.loc[0],
    binsize=0.0005,
    windowsize=[0., 0.25],
    ep=rate_nap.time_support,
)

# spike-triggered rate & auto-corr filter
plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.title("spike triggered average")
plt.plot(trig_average.t[trig_average.t > 0], trig_average.d[trig_average.t > 0]/dt_sec)
plt.ylabel("rate [Hz]")
plt.xlabel("time[sec]")
plt.subplot(122)
acg_filter = data_handler.eval_basis["spike_counts_0"][0] @ model.coef_[0,:10]
plt.plot(dt_sec + np.arange(acg_filter.shape[0]) * dt_sec, acg_filter)
plt.ylabel("a.u.")
plt.xlabel("time[sec]")
plt.tight_layout()