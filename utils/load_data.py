import warnings
try:
    from allensdk.core.cell_types_cache import CellTypesCache
except:
    warnings.warn("Allen SDK not installed! Download NWB manually and use `load_to_pynapple_without_sdk`")


import h5py
from typing import Tuple
import pynapple as nap

import numpy as np


def load_from_sdk(id_recording: int, dt_sec=0.0005):
    # Initialize a cache for the cell types database
    ctc = CellTypesCache()

    # The id you've posted seems to be a specific recording (ephys) id, so you'd use:
    data_set = ctc.get_ephys_data(id_recording)

    # print len trials
    sweap_nums = np.sort(data_set.get_sweep_numbers())
    # keep only trials of 8sec dur
    n_trials = len(sweap_nums)

    stim_trials = {}
    volt_trials = {}
    spike_counts = {}
    sweap_metadata = {}
    time_trials = {}
    spike_times = {}

    # stack stim and resp in a matrix

    for cc, num in enumerate(sweap_nums):
        # get the data for a specific trial
        dat = data_set.get_sweep(num)

        sweap_metadata[num] = data_set.get_sweep_metadata(num)["aibs_stimulus_name"]

        # get the time for each sample
        time_samp = np.arange(dat["stimulus"].shape[0]) / dat["sampling_rate"]

        # binning for spike times
        init_time = time_samp[0]
        end_time = time_samp[-1]

        time_sec = np.arange(0, int((end_time - init_time) / dt_sec) + 1) * dt_sec
        edge_spike_bin = np.hstack((time_sec, time_sec[-1] + dt_sec))

        stim_trials[num] = np.interp(time_sec, time_samp, dat["stimulus"])
        volt_trials[num] = np.interp(time_sec, time_samp, dat["response"])
        spike_counts[num] = np.histogram(
            data_set.get_spike_times(num), bins=edge_spike_bin
        )[0]
        time_trials[num] = time_sec
        spike_times[num] = data_set.get_spike_times(num)

    return (
        time_trials,
        stim_trials,
        volt_trials,
        spike_counts,
        sweap_metadata,
        spike_times,
    )


def load_to_pynapple(id_recording: int, shift_trials_by_sec=5):
    # Initialize a cache for the cell types database
    ctc = CellTypesCache()

    # The id you've posted seems to be a specific recording (ephys) id, so you'd use:
    data_set = ctc.get_ephys_data(id_recording)

    # print len trials
    sweap_nums = np.sort(data_set.get_sweep_numbers())

    # keep only trials of 8sec dur
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
        dat = data_set.get_sweep(num)
        sweep_metadata[num] = data_set.get_sweep_metadata(num)["aibs_stimulus_name"]

        time_trials.append(
            np.arange(dat["stimulus"].shape[0]) / dat["sampling_rate"] + init_trial_time
        )
        spike_times.append(np.asarray(data_set.get_spike_times(num)) + init_trial_time)
        voltage_trials.append(dat["response"])
        stim_trials.append(dat["stimulus"])

        starts.append(time_trials[-1][0])
        ends.append(time_trials[-1][-1])

        init_trial_time = shift_trials_by_sec + time_trials[-1][-1]

    trial_interval_set = nap.IntervalSet(start=starts, end=ends)
    spike_times = nap.TsGroup(
        {0: nap.Ts(t=np.hstack(spike_times))}, time_support=trial_interval_set
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


def get_sweep_metadata(sweep_number: int, fh: h5py.File):
    sweep_metadata = {}

    # the sweep level metadata is stored in
    # stimulus/presentation/Sweep_XX in the .nwb file

    # indicates which metadata fields to return
    metadata_fields = ['aibs_stimulus_amplitude_pa',
                       'aibs_stimulus_name',
                       'gain', 'initial_access_resistance', 'seal']
    try:
        stim_details = fh['stimulus']['presentation'][
            'Sweep_%d' % sweep_number]
        for field in metadata_fields:
            # check if sweep contains the specific metadata field
            if field in stim_details.keys():
                sweep_metadata[field] = stim_details[field][()]

    except KeyError:
        sweep_metadata = {}

    return sweep_metadata


def get_pipeline_version(fh: h5py.File) -> Tuple[int, int]:
    """
    Returns the AI pipeline version number, stored in the
    metadata field 'generated_by'. If that field is
    missing, version 0.0 is returned.

    Returns
    -------
    :
    The pipleline version
    """
    try:
        if 'generated_by' in fh["general"]:
            info = fh["general/generated_by"]
            # generated_by stores array of keys and values
            # keys are even numbered, corresponding values are in
            #   odd indices
            for i in range(len(info)):
                if info[i] == 'version':
                    version = info[i + 1]
                    break
        toks = version.split('.')
        if len(toks) >= 2:
            major = int(toks[0])
            minor = int(toks[1])
    except Exception:
        minor = 0
        major = 0
    return major, minor


def get_sweep(sweep_number: int, fh: h5py.File):
    """ Retrieve the stimulus, response, index_range, and sampling rate
    for a particular sweep.  This method hides the NWB file's distinction
    between a "Sweep" and an "Experiment".  An experiment is a subset of
    of a sweep that excludes the initial test pulse.  It also excludes
    any erroneous response data at the end of the sweep (usually for
    ramp sweeps, where recording was terminated mid-stimulus).

    Some sweeps do not have an experiment, so full data arrays are
    returned.  Sweeps that have an experiment return full data arrays
    (include the test pulse) with any erroneous data trimmed from the
    back of the sweep.

    Parameters
    ----------
    sweep_number:
        The sweep ID.

    fh:
        The h5py file.

    Returns
    -------
    :
        A dictionary with 'stimulus', 'response', 'index_range', and
        'sampling_rate' elements.  The index range is a 2-tuple where
        the first element indicates the end of the test pulse and the
        second index is the end of valid response data.
    """
    swp = fh['epochs']['Sweep_%d' % sweep_number]

    # fetch data from file and convert to correct SI unit
    # this operation depends on file version. early versions of
    #   the file have incorrect conversion information embedded
    #   in the nwb file and data was stored in the appropriate
    #   SI unit. For those files, return uncorrected data.
    #   For newer files (1.1 and later), apply conversion value.
    major, minor = get_pipeline_version(fh)
    if (major == 1 and minor > 0) or major > 1:
        # stimulus
        stimulus_dataset = swp['stimulus']['timeseries']['data']
        conversion = float(stimulus_dataset.attrs["conversion"])
        stimulus = stimulus_dataset[()] * conversion
        # acquisition
        response_dataset = swp['response']['timeseries']['data']
        conversion = float(response_dataset.attrs["conversion"])
        response = response_dataset[()] * conversion
    else:  # old file version
        stimulus_dataset = swp['stimulus']['timeseries']['data']
        stimulus = stimulus_dataset[()]
        response = swp['response']['timeseries']['data'][()]

    if 'unit' in stimulus_dataset.attrs:
        unit = stimulus_dataset.attrs["unit"].decode('UTF-8')

        unit_str = None
        if unit.startswith('A'):
            unit_str = "Amps"
        elif unit.startswith('V'):
            unit_str = "Volts"
        assert unit_str is not None, Exception(
            "Stimulus time series unit not recognized")
    else:
        unit = None
        unit_str = 'Unknown'

    swp_idx_start = swp['stimulus']['idx_start'][()]
    swp_length = swp['stimulus']['count'][()]

    swp_idx_stop = swp_idx_start + swp_length - 1
    sweep_index_range = (swp_idx_start, swp_idx_stop)

    # if the sweep has an experiment, extract the experiment's index
    # range
    try:
        exp = fh['epochs']['Experiment_%d' % sweep_number]
        exp_idx_start = exp['stimulus']['idx_start'][()]
        exp_length = exp['stimulus']['count'][()]
        exp_idx_stop = exp_idx_start + exp_length - 1
        experiment_index_range = (exp_idx_start, exp_idx_stop)
    except KeyError:
        # this sweep has no experiment.  return the index range of the
        # entire sweep.
        experiment_index_range = sweep_index_range

    assert sweep_index_range[0] == 0, Exception(
        "index range of the full sweep does not start at 0.")

    return {
        'stimulus': stimulus,
        'response': response,
        'stimulus_unit': unit_str,
        'index_range': experiment_index_range,
        'sampling_rate': 1.0 * swp['stimulus']['timeseries'][
            'starting_time'].attrs['rate']
    }


def get_sweep_numbers(fh):
    """ Get all of the sweep numbers in the file, including test sweeps.
    """
    sweeps = [int(e.split('_')[1])
              for e in fh['epochs'].keys() if e.startswith('Sweep_')]
    return sweeps


def get_spike_times(sweep_number, fh, key="spike_times"):
    """ Return any spike times stored in the NWB file for a sweep.

    Parameters
    ----------
    sweep_number: int
        index to access
    key : string
        label where the spike times are stored (default
        NwbDataSet.SPIKE_TIMES)

    Returns
    -------
    list
       list of spike times in seconds relative to the start of the sweep
    """
    DEPRECATED_SPIKE_TIMES = "aibs_spike_times"

    datasets = ["analysis/%s/Sweep_%d" % (key, sweep_number),
                "analysis/%s/Sweep_%d" % (
                DEPRECATED_SPIKE_TIMES, sweep_number)]

    for ds in datasets:
        if ds in fh:
            return fh[ds][()]
    return []


def load_to_pynapple_without_sdk(path: str, shift_trials_by_sec: float = 5.)\
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
    # Load the hdf5 file
    with h5py.File(path) as fh:

        # print len trials
        sweap_nums = np.sort(get_sweep_numbers(fh))

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
            dat = get_sweep(num, fh)
            sweep_metadata[num] = get_sweep_metadata(num, fh)

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
            spike_times.append(np.asarray(get_spike_times(num, fh)) + init_trial_time)

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
            {0: nap.Ts(t=np.hstack(spike_times))}, time_support=trial_interval_set
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


