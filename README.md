# TreeMazeAnalyses
## Citation
Alexander Gonzalez, Lisa M Giocomo. Parahippocampal neurons encode task-relevant information for goal-directed navigation.
eLife2024;13:RP85646 DOI: https://doi.org/10.7554/eLife.85646.4

## Pre-print
https://www.biorxiv.org/content/10.1101/2022.12.15.520660v1

## Processed Timeseries by session data on Dandi:
https://dandiarchive.org/dandiset/000405

## Additional Metadata Files on the metadata folder.

## Installation on development
1) Download zip or git clone.
2) Create a conda environment running python 3.6
'$ conda create --name new_env python=3.6'
3) activate the environment:
'$ souce activate new_env'
4) change directory to the clonned treemaze directory and install:
'$ cd TMA'
'$ python setup.py develop'
5) install requirements
'$ pip install -r requirements.txt'
6) copy notebooks outside of package, eg:
'$ cp ~Documents/TMA/Analyses/Notebooks/* ~/Documents/Notebooks/*'
7) use it! open python and try:
import TMA


8) Depending on your setup, link to a google drive for access to the data:
https://towardsdatascience.com/integrate-jupyterlab-with-google-drive-98d13e340c63
