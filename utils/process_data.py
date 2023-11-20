from typing import List, Literal, Optional

import jax

import nemos as nmo
import numpy as np
import pynapple as nap
from numpy.typing import NDArray

from utils import load_data


@jax.jit
def pytree_convolve(var_tree, eval_basis):
    """
    Convolve all trials stored in a pytree and stack them.

    Parameters
    ----------
    var_tree
    eval_basis
    filter_type

    Returns
    -------

    """

    conv_func = lambda x: nmo.utils._CORR_VARIABLE_TRIAL_DUR(x, eval_basis)
    return jax.tree_map(conv_func, var_tree)
   # return jax.numpy.vstack(jax.tree_util.tree_flatten(conv_tree)[0])


def convert_to_pynapple(time, time_support, data: Optional[NDArray] = None):
    # structure depends on the data dim
    if data is None:
        nap_obj = nap.Ts(time, time_support=time_support)
    elif data.ndim == 1:
        nap_obj = nap.Tsd(time, d=data, time_support=time_support)
    elif data.ndim == 2:
        nap_obj = nap.TsdFrame(time, d=data, time_support=time_support)
    else:
        nap_obj = nap.TsdTensor(time, d=data, time_support=time_support)
    return nap_obj


def convolve_predictor_nap(
    variable_tsd: nap.Tsd,
    eval_basis: NDArray,
    filter_type: Literal["causal", "acausal", "anti-causal"] = "causal",
):
    """
    Create a pytree representing the trial structure form pynapple. Convolve & pad, revert to pynapple.

    Parameters
    ----------
    variable_tsd
    eval_basis
    filter_type

    Returns
    -------

    """
    # create convolve pytree using pynapple get
    var_list = {
        index:
            variable_tsd.get(*variable_tsd.time_support.loc[index]).d[:, None]
        for index in variable_tsd.time_support.index
    }

    # convolve and pad to original dim using nemos
    conv_var = pytree_convolve(var_list, eval_basis)

    # revert to pynapple
    conv_var = convert_to_pynapple(variable_tsd.t, variable_tsd.time_support, conv_var)
    return conv_var


def concatenate_predictor(*predictors):
    concatenated = np.concatenate(predictors, axis=2)
    valid_entries = np.all(~np.isnan(concatenated), axis=(1, 2))
    return concatenated, valid_entries


def get_trial_key(trial_type_dict: dict, trial_label: str):
    """
    Extract the trial IDs from the keys in the trial_type_dict.

    Parameters
    ----------
    trial_type_dict:
        Dictionary
    trial_label

    Returns
    -------

    """
    if not type(trial_label) is bytes:
        trial_label = trial_label.encode("utf-8")
    return [
        key for key, value in trial_type_dict.items() if value == trial_label
    ]


def create_pytree(var_name, nap_variable, ep=None):
    if ep is None:
        ep = nap_variable.time_support

    # create convolve pytree using pynapple get
    var_tree = {
        index:
            nap_variable.get(*ep.loc[index]).d[:, None]
        for index in ep.index
    }
    return {var_name: var_tree}


def add_inputs_to_pytree(var_dict, var_name, nap_variable, ep=None):
    if ep is None:
        ep = nap_variable.time_support
    # add input to pytree
    var_dict.update(create_pytree(var_name, nap_variable, ep=ep))


class PynappleLoader:
    def __init__(self, specimen_id, bin_size_sec=0.0005, loc=0., scale=1.):
        (
            self.trial_support,
            self.injected_current,
            self.voltages,
            self.spike_times,
            self.sweap_metadata,
        ) = load_data.load_to_pynapple(specimen_id)
        self.bin_size_sec = bin_size_sec
        self.predictor_dict = {}
        self.spike_counts_dict = {}
        self._filter_trials = tuple()
        self.loc = 0.
        self.scale = np.max(self.injected_current)


    @property
    def filter_trials(self):
        return self._filter_trials

    @filter_trials.setter
    def filter_trials(self, trial_groups):
        for label in trial_groups:
            if label in self.sweap_metadata.keys():
                raise NameError(f"`{label}` is not a trial group.")
        self._filter_trials = trial_groups
        print("updating predictors...")
        self.create_predictors(*self.predictor_dict.keys())

    def set_trial_filter(self, *trial_groups):
        self.filter_trials = trial_groups

    def get_trial_key(self, trial_label: str):
        if not type(trial_label) is bytes:
            trial_label = trial_label.encode("utf-8")
        return [
            key for key, value in self.sweap_metadata.items() if value == trial_label
        ]

    def clear_variables(self):
        self.predictor_dict = {}
        self.spike_counts_dict = {}

    def get_trial_types(self, *trial_labels):
        trial_list = []
        for label in trial_labels:
            trial_list += self.get_trial_key(label)
        return np.sort(trial_list)

    def add_counts(self, unit: int):
        trial_ids = self.get_trial_types(*self.filter_trials)
        spike_counts = self.spike_times.count(
            bin_size=self.bin_size_sec, ep=self.trial_support.loc[trial_ids]
        )

        add_inputs_to_pytree(
            self.predictor_dict,
            var_name=f"spike_counts_{unit}",
            nap_variable=spike_counts[:, unit]
        )

    def add_stimuli(
            self,
            scale: Optional[float] = None,
            loc: Optional[float] = None
    ):
        if not loc is None:
            self.loc = loc

        if not scale is None:
            self.scale = scale

        scaled_stim = (self.injected_current - self.loc) / self.scale
        trial_ids = self.get_trial_types(*self.filter_trials)
        stimuli = scaled_stim.bin_average(
            bin_size=self.bin_size_sec, ep=self.trial_support.loc[trial_ids]
        )
        add_inputs_to_pytree(
            self.predictor_dict,
            var_name="injected_current",
            nap_variable=stimuli
        )

    def bin_spikes(self):
        trial_ids = self.get_trial_types(*self.filter_trials)
        spike_counts = self.spike_times.count(
            bin_size=self.bin_size_sec, ep=self.trial_support.loc[trial_ids]
        )
        for unit in range(spike_counts.shape[1]):
            add_inputs_to_pytree(self.spike_counts_dict, f"spike_counts_{unit}", spike_counts)

    def create_predictors(self, *predictors_names):
        for name in predictors_names:
            if (name != "injected_current") and not name.startswith("spike_counts_"):
                raise ValueError(f"Unknown predictor '{name}'. Choose between [spike_counts_i, injected_current].")

        if predictors_names:
            self.clear_variables()
            for name in predictors_names:
                if name.startswith("spike_counts_"):
                    self.add_counts(int(name.split("spike_counts_")[1]))
                elif name == "injected_current":
                    self.add_stimuli()
            self.bin_spikes()


class ModelDataHandler:
    def __init__(self, predictor_dict: dict, counts_dict: dict, bin_size_sec=0.0005):

        self.eval_basis = {}

        self.predictor_dict = predictor_dict
        self.counts_dict = counts_dict

        self.bin_size_sec = bin_size_sec

    def clear_basis(self):
        self.eval_basis = {}

    def set_basis(self, var_name, basis, window_size_ms, basis_kwargs, filter_type="causal"):
        window_size = int(window_size_ms / (1000 * self.bin_size_sec))
        self.eval_basis[var_name] = basis(**basis_kwargs).evaluate(
            np.linspace(0, 1, window_size)
        ), filter_type


    # def set_basis_acg(self, basis, window_size_ms, basis_kwargs):
    #     window_size = int(window_size_ms / (1000 * self.bin_size_sec))
    #     self.eval_basis_acg = basis(**basis_kwargs).evaluate(
    #         np.linspace(0, 1, window_size)
    #     )
    #
    # def get_trial_key(self, trial_label: str):
    #     if not type(trial_label) is bytes:
    #         trial_label = trial_label.encode("utf-8")
    #     return [
    #         key for key, value in self.sweap_metadata.items() if value == trial_label
    #     ]
    #
    # def get_trial_types(self, *trial_labels):
    #     trial_list = []
    #     for label in trial_labels:
    #         trial_list += self.get_trial_key(label)
    #     return np.sort(trial_list)
    #
    # def subsample_trials(self, var_dict, *trial_labels):
    #     trial_list = self.get_trial_types(*trial_labels)
    #     return [var_dict[trial] for trial in trial_list]
    #
    # def scale_stim(self):
    #     scaled_stimulus = (self.nap_stimulus - self.loc) / self.scale
    #     return scaled_stimulus

    def get_convolved_predictor(self):
        # spike_counts = self.nap_spike_times.count(
        #     bin_size=self.bin_size_sec, ep=self.nap_trials.loc[trial_ids]
        # )
        # scaled_stim = self.scale_stim().bin_average(
        #     bin_size=self.bin_size_sec, ep=self.nap_trials.loc[trial_ids]
        # )
        # stim = convolve_predictor_nap(
        #     scaled_stim,
        #     self.eval_basis_stim,
        #     filter_type="causal"
        # )
        # conv_spk = convolve_predictor_nap(
        #     spike_counts[:, 0],
        #     self.eval_basis_acg,
        #     filter_type="causal",
        # )
        conv_list = []
        for key in self.eval_basis:
            window_size = self.eval_basis[key][0].shape[1]
            filter_type = self.eval_basis[key][1]
            conv_tree = pytree_convolve(self.predictor_dict[key], self.eval_basis[key][0])
            pad_func = lambda x: nmo.utils._pad_dimension(
                x, 1, window_size, filter_type, constant_values=jax.numpy.nan
            )
            conv_list.append(
                jax.numpy.vstack(
                    jax.tree_util.tree_flatten(jax.tree_util.tree_map(pad_func, conv_tree))[0]
                )
            )
        X, valid = concatenate_predictor(*conv_list)
        y = jax.numpy.vstack(jax.tree_util.tree_flatten(self.counts_dict)[0])

        # X = nap.TsdTensor(
        #     t=conv_spk.t[valid], d=X[valid], time_support=conv_spk.time_support
        # )
        # y = nap.TsdFrame(
        #     t=conv_spk.t[valid], d=y[valid], time_support=conv_spk.time_support
        # )
        return X, y
