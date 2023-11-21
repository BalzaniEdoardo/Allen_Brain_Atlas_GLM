from pathlib import Path

import numpy as np
import pynapple as nap
from allensdk.core.cell_types_cache import CellTypesCache


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

