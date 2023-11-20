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
specimen_id = 609492577

# parameters
train_trial_labels = "Long Square", "Short Square"
window_size_acg = 250
window_size_stim = 200
dt_sec = 0.0005
n_basis_acg = 10
n_basis_stim = 10


nap_experiment = process_data.PynappleLoader(specimen_id)
nap_experiment.set_trial_filter(*train_trial_labels)

# create nested dictionary with predictors, first key: "predictor name". second key: "trial ID"
nap_experiment.create_predictors("spike_counts_0", "injected_current")
data_handler = process_data.ModelDataHandler(
        predictor_dict=nap_experiment.predictor_dict,
        counts_dict=nap_experiment.spike_counts_dict,
        bin_size_sec=nap_experiment.bin_size_sec
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
X, y = data_handler.get_convolved_predictor()
# nap_experiment.add_convolved_spike_history()
# nap_experiment.add_stimuli()
# nap_experiment.bin_spikes()

#proc = process_data.ModelDataHandler(predictor_dict=obj.predictor_dict, counts_dict=obj.spike_counts_dict, bin_size_sec=obj.bin_size_sec)



# # process data: This will download and cache the recordings (it will take a while the first time
# # you analyze a cell, each file is tens of MB).
# proc = process_data.TrialHandlerAllenPynapple(specimen_id)
#
# print("stimulation type is stored in proc.sweap_metadata:")
# print(proc.sweap_metadata)
#
# proc.set_basis_stim(
#     nmo.basis.RaisedCosineBasisLog, window_size_stim, dict(n_basis_funcs=n_basis_stim)
# )
#
# proc.set_basis_acg(
#     nmo.basis.RaisedCosineBasisLog, window_size_acg, dict(n_basis_funcs=n_basis_acg + 1)
# )
# # remove the last basis (which is the first in the basis matrix)
# proc.eval_basis_acg = proc.eval_basis_acg[:, 1:]


# # create design matrix by convolving stimulus and spikes with the basis
# X, y = proc.get_convolved_predictor(*train_trial_labels)
#
# # Model definition
# regularizer_strength = 8 * 10**-11
# solver = nmo.solver.RidgeSolver(
#     "GradientDescent",
#     solver_kwargs={"jit": True},
#     regularizer_strength=regularizer_strength,
# )
# obs_model = nmo.observation_models.PoissonObservations(
#     inverse_link_function=jax.nn.softplus
# )
# model = nmo.glm.GLMRecurrent(solver=solver, observation_model=obs_model)
#
# # initialize and fit the model
# init_params = np.zeros((1, n_basis_stim + n_basis_acg)), np.log(np.mean(y, axis=0))
# model.fit(X, y, init_params=init_params)
#
# # predict the rate
# rate = nap.TsdFrame(t=y.t, d=np.asarray(model.predict(X)), time_support=y.time_support)
# trig_average = nap.compute_event_trigger_average(
#     proc.nap_spike_times,
#     rate.loc[0],
#     binsize=0.0005,
#     windowsize=[0., 0.3],
#     ep=rate.time_support,
# )
#
# acg_raw = nap.compute_autocorrelogram(proc.nap_spike_times, binsize=0.001, windowsize=1, norm=False)
# plt.plot(trig_average.t, trig_average.d[trig_average.t>0])