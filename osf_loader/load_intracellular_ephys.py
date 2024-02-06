import numpy as np
import pynapple as nap
from allensdk.core.cell_types_cache import CellTypesCache
from typing import Tuple
import h5py


def find_keys(dict_hdf5, target, key, list_key):

    if isinstance(key, bytes):
        key = key.decode()

    if not isinstance(dict_hdf5[key], dict):
        return False

    for key_nest in dict_hdf5[key]:

        if isinstance(key_nest, bytes):
            key_nest = key_nest.decode()

        if target in key_nest:
            return key_nest

        return False


def nested_find_keys(dict_hdf5, target, key, list_key=None):
    if list_key is None:
        list_key = []

    found = find_keys(dict_hdf5, target, key, list_key)
    if found:
        list_key.append(found)
        return list_key

    elif isinstance(dict_hdf5[key], dict):
        for key_nest in dict_hdf5[key].keys():
            list_key = nested_find_keys(dict_hdf5[key], target, key_nest, list_key)
            if len(list_key):
                list_key = [key_nest] + list_key
                break

    return list_key


def read_hdf5_dataset(dataset):
    # Read the dataset
    data = dataset[()]

    # Check the data type and handle accordingly
    if isinstance(data, bytes):  # Handle strings
        return data.decode()  # Convert bytes to a string
    elif isinstance(data, np.ndarray):  # Handle arrays
        return data  # Return the array as is
    else:
        #print(f"new type {type(data)}")
        return data  # For other data types, return as is


def unpack_group(hdf5_group, root_name, unpacked=None):
    """
    Recursive call to unpack hdf5.

    Parameters
    ----------
    hdf5_fh
    root_name
    unpacked

    Returns
    -------

    """
    if unpacked is None:
        unpacked = {}

    if isinstance(hdf5_group[root_name], h5py.Group):
        if len(hdf5_group[root_name]) == 0:
            return unpacked
        unpacked[root_name] = {}
        for name in hdf5_group[root_name].keys():
            unpack_group(hdf5_group[root_name], name, unpacked[root_name])

    elif isinstance(hdf5_group[root_name], h5py.Dataset):
        unpacked[root_name] = read_hdf5_dataset(hdf5_group[root_name])
    else:
        raise TypeError(f"Unrecognized type {type(hdf5_group[root_name])}")

    return unpacked


def load_to_pynapple(specimen_id: int, shift_trials_by_sec: float = 5.)\
        -> Tuple[nap.IntervalSet, nap.Tsd, nap.Tsd, nap.TsGroup, dict]:
    """
    Load the intracellular recording as pynapple time series.

    Parameters
    ----------
    specimen_id:
        The id of the specimen as in the allen brain map experiment.

        https://celltypes.brain-map.org/experiment/electrophysiology/
    shift_trials_by_sec:
        Shift used to concatenate trials in seconds. In the original dataset,
        every trial starts at t=0.
        Pynapple time index must be monotonically increasing
        instead. We artificially add a time shift to each trial in order
        to create a consistent, monotonic time index.

    Returns
    -------
    trial_interval_set
        The interval set containing the trial start and end in seconds.
    stim_trials
        The injected current time series.
    voltages
        The sub-threshold voltages.
    spike_times
        The spike times in seconds.
    sweep_metadata
        Metadata describing the stimulation protocol.
    """
    # Initialize a cache for the cell types database
    ctc = CellTypesCache()

    # The id you've posted seems to be a specific recording (ephys) id, so you'd use:
    dataset = ctc.get_ephys_data(specimen_id)

    # print len trials
    sweap_nums = np.sort(dataset.get_sweep_numbers())

    # Initialize the objects that will be used to construct
    # the pynapple timeseries.
    init_trial_time = 0
    sweep_metadata = {}
    stim_trials = []
    voltage_trials = []
    spike_times = []
    time_trials = []
    starts = []
    ends = []
    for cc, num in enumerate(sweap_nums):
        # get the data for a specific trial
        dat = dataset.get_sweep(num)
        sweep_metadata[num] = dataset.get_sweep_metadata(num)

        # append metadata information
        sweep_metadata[num].update(
            {
                "stimulus_unit": dat["stimulus_unit"],
                "sampling_rate": dat["sampling_rate"],
                "response_unit": "Volt"
            }
        )

        # compute the time index for the trial by dividing the number of
        # samples by the sampling rate and adding a time shift that
        # guarantees that the time index is strictly increasing.
        time_trials.append(
            np.arange(dat["stimulus"].shape[0]) / dat["sampling_rate"] + init_trial_time
        )
        # add the same time shift to the spike times of the trial
        spike_times.append(np.asarray(dataset.get_spike_times(num)) + init_trial_time)

        # append voltage and injected current
        voltage_trials.append(dat["response"])
        stim_trials.append(dat["stimulus"])

        # store the first and last timestamp of each trial
        starts.append(time_trials[-1][0])
        ends.append(time_trials[-1][-1])

        # compute the next time shift
        init_trial_time = shift_trials_by_sec + time_trials[-1][-1]

    # define the pynapple objects
    trial_interval_set = nap.IntervalSet(start=starts, end=ends)
    spike_times = nap.TsGroup(
        {1: nap.Ts(t=np.hstack(spike_times))}, time_support=trial_interval_set
    )
    voltages = nap.Tsd(
        t=np.hstack(time_trials),
        d=np.hstack(voltage_trials),
        time_support=trial_interval_set,
    )
    stim_trials = nap.Tsd(
        t=voltages.t, d=np.hstack(stim_trials), time_support=trial_interval_set
    )

    return trial_interval_set, stim_trials, voltages, spike_times, sweep_metadata


if __name__ == "__main__":
    fh = h5py.File("../cell_types/specimen_609492577/ephys.nwb", "r")

    # get rate
    #fh['epochs']['Sweep_10']['stimulus']['timeseries']["starting_time"].attrs['rate']
    for key in fh.keys():
        unpacked = unpack_group(fh, key)
        for key_nest in unpacked.keys():
            print(key, nested_find_keys(unpacked, "rate", key_nest))