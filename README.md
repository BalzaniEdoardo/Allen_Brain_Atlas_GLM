# Allen_Brain_Atlas_GLM
Fit a Nemos GLM to the Allen Brain Atlas intracellular recordings.

## Warining
The API of Nemos, and, to a lesser extent, of pynapple are changing. This may results in breaking changes to the `fit_allen_brain_atlas.py` script. Reach out to me if this happens.

## Installation

Activate the desired python enviroment. Cd to the folder in which you cloned or downloaded this repo. Install the requirements with

```bash
pip install -r requirements.txt
```

The requirements **do not** include the Allen-SDK  package. 

## Getting the data
To find a specimen you can browse the `cell_types_specimen_details.csv`. This can be loded in `pandas` and explored.


```python
import pandas as pd
from utils import process_data
info_recordings = pd.read_csv("path-to-file/cell_types_specimen_details.csv")
# select some id
specimen_id = info_recordings["specimen__id"][0]
```

The Allen-SDK is being deprecated, if you are able to install the Allen-SDK, congratulations! You can now download any data from the Allen-SDK by simply selecting any specimen-ID in the "cell_types_specimen_details.csv" table and run the following lines of code in python.

```python
from utils import process_data

# select binning for downsampling voltages and current (original sampling rate 200KHz)
bin_size_sec = 0.001

# load data into pynapple
experiment = process_data.PynappleLoader(specimen_id, bin_size_sec=bin_size_sec)
```

If you are not able to install the Allen-SDK, you can still download the file by browsing to the brain atlas website, and click the download link. 

In order to find the `specimen_id`, as before load `cell_types_specimen_details.csv` and explore the content, then browse to:

https://celltypes.brain-map.org/experiment/electrophysiology/your-specimen-id

where you should substitute the *your-specimen-id* with the ID number you found in the `cell_types_specimen_details.csv`.

Download the data in a path that contains the specimen ID. If you want to respect the SDK folder structure save the file as `ephys.nwb`, in the follwoing folder:


```*local-path*/cell_types/specimen_*specimen_id*/ephys.nwb```

Once you have the file you can run python and load the data with:
```python
from utils import process_data

# select binning for downsampling voltages and current (original sampling rate 200KHz)
bin_size_sec = 0.001

# load data into pynapple
experiment = process_data.PynappleLoader(
  specimen_id,
  bin_size_sec=bin_size_sec,
  path="local-path/cell_types/specimen_%d/ephys.nwb"
)
```

## Fitting the model

Run the script `fit_allen_brain_atlas.py`. If you have the SDK installed, you can run a fit just by setting the specimen ID variable. If you don't
you first need to download the data manually, then change the line

```
experiment = process_data.PynappleLoader(specimen_id, bin_size_sec=dt_sec)
```

to

```
experiment = process_data.PynappleLoader(specimen_id, bin_size_sec=dt_sec, path="local-path/cell_types/specimen_%d/ephys.nwb")
```

And modify the path to your local path to the file.


