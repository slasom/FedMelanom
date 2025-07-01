# evaluations/

This directory is intended to store results and logs generated during development and testing by the developer.  
It is designed to hold model evaluation outputs that may be produced using the auxiliary functions provided in the API.  
The use of this folder is optional and it may not be present in all deployments.

Each subfolder within `evaluations/` corresponds to a specific evaluation run, using an independent dataset for each model evaluated.  
Within the `train` and `retrain` subfolders, you will find timestamped files containing the metrics and additional results obtained during the respective training or retraining processes.
