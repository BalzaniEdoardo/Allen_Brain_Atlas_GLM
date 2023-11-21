import jax
import matplotlib.pylab as plt
import neurostatslib as nmo
import numpy as np
import pandas as pd
import pynapple as nap

from utils import load_data, process_data

# Load the recording information
info_recordings = pd.read_csv("cell_types_specimen_details.csv")
print(info_recordings.loc[
        2172,
        ['line_name', 'specimen__id', 'specimen__name',
         'structure__id', 'structure__name', 'structure__acronym']
])


# select a recording
specimen_id = 609492577



# Parameters
train_trial_labels = "Long Square", "Short Square"
window_size_acg = 125   # ms
window_size_stim = 125  # ms
dt_sec = 0.0005
n_basis_acg = 13
n_basis_stim = 13

# load the experiment (creates pynapple objects)
experiment = process_data.PynappleLoader(specimen_id, bin_size_sec=dt_sec)

# select some stimulation protocols
experiment.set_trial_filter(*train_trial_labels)

# create nested dictionary with predictors, first key: "predictor name". second key: "trial ID"
experiment.construct_predictors_and_counts_pytree("spike_counts_0", "injected_current")

# Plot the whole recording
norm_stim = experiment.injected_current.bin_average(bin_size=0.001) / np.max(experiment.injected_current)
fig, ax = plt.subplots(figsize=(10, 3.5))
plt.plot(norm_stim, label="stimulus")
plt.vlines(experiment.spike_times[1].t, 0, 0.2, 'k', label="spikes")
ylim = plt.ylim()
for start, end in experiment.spike_times.time_support.values:
    if start == 0:
        rect = plt.Rectangle(
            xy=(start, 0),
            width=end - start,
            height=ylim[1],
            alpha=0.2,
            facecolor="grey",
            label="trial"
        )
    else:
        rect = plt.Rectangle(
            xy=(start, 0),
            width=end - start,
            height=ylim[1],
            alpha=0.2,
            facecolor="grey"
        )
    ax.add_patch(rect)
plt.xlabel('time [sec]')
plt.ylabel('a.u.')
plt.legend()

# add the predictors and the count to object that will
# construct the model design matrix
# the bin size is used to determine the basis window size
model_constructor = process_data.ModelConstructor(
        predictor_dict=experiment.predictor_dict,
        counts_dict=experiment.spike_counts_dict,
        bin_size_sec=experiment.bin_size_sec
)
model_constructor.set_basis(
        "spike_counts_0",
        nmo.basis.RaisedCosineBasisLog,
        window_size_acg,
        dict(n_basis_funcs=n_basis_acg, alpha=2.5, clip_first=False)
)
model_constructor.set_basis(
        "injected_current",
        nmo.basis.RaisedCosineBasisLog,
        window_size_stim,
        dict(n_basis_funcs=n_basis_stim, alpha=2.5, clip_first=False)
)


X, y, valid = model_constructor.get_convolved_predictor()

# Plot the convolution output
time = experiment.get_time()
conv_out = nap.TsdFrame(t=time, d=X[:, 0], time_support=experiment.trial_support).dropna()
fig, ax = plt.subplots(figsize=(10, 3.5))
plt.plot(norm_stim, label="stimulus")
plt.plot(conv_out[:, :5])
plt.vlines(experiment.spike_times[1].t, 0, 1., 'k', label="spikes")
plt.xlabel('time [sec]')
plt.ylabel('a.u.')
plt.legend()

# Model definition
regularizer_strength = 8 * 10**-11
solver = nmo.solver.RidgeSolver(
    "GradientDescent",
    solver_kwargs={"jit": True},
    regularizer_strength=regularizer_strength,
)
obs_model = nmo.noise_model.PoissonNoiseModel(
    inverse_link_function=jax.nn.softplus
)
model = nmo.glm.GLMRecurrent(solver=solver, noise_model=obs_model)

# initialize and fit the model
init_params = np.zeros((1, n_basis_stim + n_basis_acg)), np.log(np.mean(y, axis=0))
model.fit(X[valid], y[valid], init_params=init_params)
rate = model.predict(X)

# convert back to pynapple
time = experiment.get_time()
rate_nap = nap.TsdFrame(
    t=time,
    d=np.asarray(rate),
    time_support=experiment.spike_times.time_support
).dropna()


trig_average_raw = nap.compute_event_trigger_average(
    experiment.spike_times,
    experiment.spike_times[1].count(bin_size=0.0005),
    binsize=0.0005,
    windowsize=[0., 0.25],
    ep=rate_nap.time_support,
)

trig_average = nap.compute_event_trigger_average(
    experiment.spike_times,
    rate_nap.loc[0],
    binsize=0.0005,
    windowsize=[0.001, 0.25],
    ep=rate_nap.time_support,
)




# spike-triggered rate & auto-corr filter
plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.title("auto-correlogram")
plt.plot(trig_average_raw.t[trig_average_raw.t > 0.0006],
         trig_average_raw.d[trig_average_raw.t > 0.0006]/dt_sec, label="raw")
plt.plot(trig_average.t[trig_average.t > 0.0006],
         trig_average.d[trig_average.t > 0.0006]/dt_sec, label="model")
plt.ylabel("rate [Hz]")
plt.xlabel("time[sec]")
plt.legend()
plt.subplot(122)
acg_filter = model_constructor.eval_basis["spike_counts_0"] @ model.basis_coeff_[0, :n_basis_acg]
plt.plot(dt_sec + np.arange(acg_filter.shape[0]) * dt_sec, acg_filter)
plt.ylabel("a.u.")
plt.xlabel("time[sec]")
plt.tight_layout()

