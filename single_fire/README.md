
## Overview

Use pick_arrivals.py to add P and S phase labels for EQ and EX data from Maguire et al., 2024. 
It uses the PhaseNet model trained with STEAD from SeisBench to identify the arrivals. If there are no detected arrivals, use the predicted arrival time using average P and S group velocities. 

Run for a single experiment/source type using command line and appropriate arguments. e.g.,
```
python pick_arrivals.py -e "base" -s "ex" -o "."
```
- See all argument options using `python pick_arrivals.py -h`
  
loop_pick_arrivals.py contains an example of how to loop over experiments and/or sources.

## Dependencies
I followed the [cpu only instructions from the SeisBench](https://github.com/seisbench/seisbench?tab=readme-ov-file#cpu-only-installation) 

- Info:
  - torch install command I used: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
  - versions:
    - python: 3.11.9
    - pytorch: 2.4.1
    - torchaudio: 2.4.1
    - torchvision: 0.19.1
    - seisbench: 0.8.0