from typing import Optional

import jax

import neurostatslib as nmo
import numpy as np
import pynapple as nap
from numpy.typing import NDArray

from utils import load_data


@jax.jit
def pytree_convolve(var_tree, eval_basis):
    """
    Applies a convolution operation to each element of a PyTree and adds a dimension to the result.

    Parameters
    ----------
    var_tree : PyTree
        A PyTree containing the data to be convolved.
    eval_basis : array_like
        The basis function to be used for convolution.

    Returns
    -------
    PyTree
        A new PyTree with each element being the convolution of the corresponding element in `var_tree` with `eval_basis`.
    """
    conv_func = lambda x: nmo.utils._CORR_VARIABLE_TRIAL_DUR(x, eval_basis)[None]
    return jax.tree_map(conv_func, var_tree)


def convert_to_pynapple(time, time_support, data: Optional[NDArray] = None):
    """
    Converts time series data into a Pynapple data structure.

    Parameters
    ----------
    time : array_like
        Time points of the data.
    time_support : array_like
        The support of the time points.
    data : NDArray, optional
        The data to be converted. If None, only time points are converted.

    Returns
    -------
    object
        A Pynapple time series object corresponding to the input data.
    """
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


def concatenate_predictor(*predictors):
    """
    Concatenates multiple predictor arrays along the last axis and identifies valid entries.

    Parameters
    ----------
    *predictors : array_like
        Variable number of predictor arrays to be concatenated.

    Returns
    -------
    tuple of ndarray
        A tuple containing the concatenated predictor array and a boolean array indicating valid entries.
    """
    concatenated = np.concatenate(predictors, axis=2)
    valid_entries = np.all(~np.isnan(concatenated), axis=(1, 2))
    return concatenated, valid_entries


def create_pytree(var_name, nap_variable, ep=None):
    """
    Creates a PyTree structure from a Pynapple variable.

    Parameters
    ----------
    var_name : str
        The name of the variable to be included in the PyTree.
    nap_variable : Pynapple object
        The Pynapple variable to be converted into a PyTree.
    ep : array_like, optional
        The epoch over which to create the PyTree. If None, the time support of the `nap_variable` is used.

    Returns
    -------
    dict
        A PyTree structure with the given variable name and data.
    """
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
    """
    Adds a new variable to an existing PyTree.

    Parameters
    ----------
    var_dict : dict
        The PyTree to which the new variable will be added.
    var_name : str
        The name of the variable to be added.
    nap_variable : Pynapple object
        The Pynapple variable to be added to the PyTree.
    ep : array_like, optional
        The epoch over which to add the variable. If None, the time support of the `nap_variable` is used.
    """
    if ep is None:
        ep = nap_variable.time_support
    # add input to pytree
    var_dict.update(create_pytree(var_name, nap_variable, ep=ep))


class PynappleLoader:
    """
    Loader class for handling and processing data using the Pynapple library.

    This class is designed to load, filter, and prepare data for analysis,
    specifically tailored for time series and spike count data.

    Parameters
    ----------
    specimen_id : str
        Identifier for the data specimen.
    bin_size_sec : float, optional
        Bin size in seconds for time series data.
    loc : float, optional
        Location parameter for data normalization.
    scale : float, optional
        Scale parameter for data normalization.

    Attributes
    ----------
    trial_support : nap.IntervalSet
        The support structure of trials.
    injected_current : nap.Tsd
        Time series data representing injected current.
    voltages : nap.Tsd
        Voltage data.
    spike_times : nap.TsGroup
        Spike timing data.
    sweep_metadata : dict
        Metadata for data sweeps.
    bin_size_sec : float
        Bin size used for binning spike times.
    predictor_dict : dict
        Dictionary to store predictor variables.
    spike_counts_dict : dict
        Dictionary to store binned spike counts.
    _filter_trials : tuple
        Tuple of trial group labels for filtering data.
    loc : float
        Location parameter for data normalization.
    scale : float
        Scale parameter for data normalization.
    """
    def __init__(self, specimen_id, bin_size_sec=0.0005, loc=0., scale=1.):
        (
            self.trial_support,
            self.injected_current,
            self.voltages,
            self.spike_times,
            self.sweep_metadata,
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
            if label in self.sweep_metadata.keys():
                raise NameError(f"`{label}` is not a trial group.")
        self._filter_trials = trial_groups
        print("updating predictors...")
        self.construct_predictors_and_counts_pytree(*self.predictor_dict.keys())

    def set_trial_filter(self, *trial_groups):
        """
        Sets the trial filter to specified trial groups.

        Parameters
        ----------
        *trial_groups : str
            Variable number of trial group labels to be used for filtering.
        """
        self.filter_trials = trial_groups

    def get_trial_key(self, trial_label: str):
        """
        Retrieves trial keys based on a specific trial label.

        Parameters
        ----------
        trial_label : str
            The label of the trial to retrieve keys for.

        Returns
        -------
        list:
            List of trial keys corresponding to the specified label.
        """

        if not type(trial_label) is bytes:
            trial_label = trial_label.encode("utf-8")
        return [
            key for key, value in self.sweep_metadata.items() if value == trial_label
        ]

    def clear_variables(self):
        """
        Clears the stored predictor and spike counts variables.
        """
        self.predictor_dict = {}
        self.spike_counts_dict = {}

    def get_trial_types(self, *trial_labels):
        """
        Retrieves trial types for given trial labels.

        Parameters
        ----------
        *trial_labels : str
            Variable number of trial labels to retrieve types for.

        Returns
        -------
        ndarray
            Array of sorted trial types corresponding to the specified labels.
        """
        trial_list = []
        for label in trial_labels:
            trial_list += self.get_trial_key(label)
        return np.sort(trial_list)

    def add_counts(self, unit: int):
        """
        Adds spike count data for a specific unit to the predictor dictionary.

        Parameters
        ----------
        unit : int
            The unit number for which spike counts are to be added.
        """
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
        """
        Adds stimuli data to the predictor dictionary, optionally applying scaling and location parameters.

        Parameters
        ----------
        scale : float, optional
            Scale factor to apply to the stimuli data.
        loc : float, optional
            Location parameter to apply to the stimuli data.
        """
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
        """
        Bins spike data across all units and stores them in the spike counts dictionary.
        """
        trial_ids = self.get_trial_types(*self.filter_trials)
        spike_counts = self.spike_times.count(
            bin_size=self.bin_size_sec, ep=self.trial_support.loc[trial_ids]
        )
        for unit in range(spike_counts.shape[1]):
            add_inputs_to_pytree(self.spike_counts_dict, f"spike_counts_{unit}", spike_counts[:, unit])

    def construct_predictors_and_counts_pytree(self, *predictors_names):
        """
        Creates predictor variables based on specified names and adds them to the predictor and spike count
         dictionaries.

        Parameters
        ----------
        *predictors_names : str
            Names of the predictors to be created.
        """
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

    def get_time(self):
        """
        Compute the time index vector for a given bin size. This can be done in a more
        principled way.
        """
        trial_ids = self.get_trial_types(*self.filter_trials)
        return self.spike_times.count(
            bin_size=self.bin_size_sec, ep=self.trial_support.loc[trial_ids]
        ).t


class ModelConstructor:
    """
    A class for constructing and managing the model design matrix with convolved predictors and spike count data.

    This class handles the creation of evaluation bases for variables, convolution of predictors,
    and preparation of input and output matrices for model fitting.

    Parameters
    ----------
    predictor_dict : dict
       A dictionary of predictor variables.
    counts_dict : dict
       A dictionary containing spike counts data.
    bin_size_sec : float, optional
       The bin size in seconds used for spike count data.

    Attributes
    ----------
    eval_basis : dict
       A dictionary to store evaluation bases for variables.
    predictor_dict : dict
       A dictionary of predictor variables.
    counts_dict : dict
       A dictionary containing spike counts data.
    bin_size_sec : float
       The bin size in seconds used for spike count data.
    """
    def __init__(self, predictor_dict: dict, counts_dict: dict, bin_size_sec=0.0005):

        self.eval_basis = {}
        self.filter_type = {}

        self.predictor_dict = predictor_dict
        self.counts_dict = counts_dict

        self.bin_size_sec = bin_size_sec

    def clear_basis(self):
        """
        Clears the stored evaluation bases and filter type.
        """
        self.eval_basis = {}
        self.filter_type = {}

    def set_basis(self, var_name, basis, window_size_ms, basis_kwargs, filter_type="causal", remove_last=True):
        """
        Sets the evaluation basis for a specified variable.

        Parameters
        ----------
        var_name : str
            The name of the variable for which to set the basis.
        basis : function
            The basis function to be used.
        window_size_ms : int
            The window size in milliseconds for the basis function.
        basis_kwargs : dict
            Additional keyword arguments for the basis function.
        filter_type : str, optional
            The type of filter to be used, default is 'causal'.
        remove_last: bool,
            Remove the last element of the basis
        """
        window_size = int(window_size_ms / (1000 * self.bin_size_sec))
        if remove_last:
            basis_kwargs["n_basis_funcs"] += 1
            self.eval_basis[var_name] = basis(**basis_kwargs).evaluate(
                np.linspace(0, 1, window_size)
            )
            self.eval_basis[var_name] = self.eval_basis[var_name][:, :-1]
        else:
            self.eval_basis[var_name] = basis(**basis_kwargs).evaluate(
                np.linspace(0, 1, window_size)
            )
        self.filter_type[var_name] = filter_type

    def get_convolved_predictor(self):
        """
        Computes and returns the convolved predictors, the spike counts, and a validity mask.

        This method performs convolution on the predictors, prepares the spike count data, and
        computes a validity mask indicating valid entries in the convolved predictors.

        Returns
        -------
        X : ndarray
            A matrix of convolved predictors. This matrix is constructed by convolving each predictor
            with its respective basis and concatenating the results.

        y : ndarray
            An array of spike counts corresponding to the convolved predictors. This array is prepared
            based on the counts data available in the counts_dict.

        valid : ndarray
            A validity mask as a boolean array, indicating which entries in the convolved predictors are valid.
            This mask is useful for filtering out invalid or missing data points in subsequent analyses.
        """
        conv_list = []
        for key in self.eval_basis:
            # get the window size and the filter type
            window_size = self.eval_basis[key].shape[0]
            filter_type = self.filter_type[key]

            # apply the convolution over the whole the pytree
            conv_tree = pytree_convolve(self.predictor_dict[key], self.eval_basis[key])

            # pad to match original size
            pad_func = lambda x: nmo.utils.nan_pad_conv(x, window_size, filter_type)

            # flatten the tree then stack over the time axis
            conv_list.append(
                jax.numpy.vstack(
                    jax.tree_util.tree_flatten(jax.tree_util.tree_map(pad_func, conv_tree))[0]
                )
            )
        # concatenate the predictor over the feature axis
        X, valid = concatenate_predictor(*conv_list)

        # stack the counts over the time axis
        y = jax.numpy.vstack(jax.tree_util.tree_flatten(self.counts_dict)[0])
        return X, y, valid
