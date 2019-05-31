# Machine Learning with DonorsChoose


## Getting Started

Conda has been used as the package manager for this project.

### Prerequisites

1. Clone the repository.

        $ git clone https://github.com/capp-machine-learning/DonorsChoose-machine-learning.git
        $ cd DonorsChoose-machine-learning

1. A yml file of the environment is available in `environment.yml`.

        $ conda env create --file=environment.yml
        $ conda activate donorsml
        
### Files

To replicate the pipeline process of this project, please make sure that all packages in `environment.yml` are __correctly installed__ and run the following commands.

        $ chmod +x ./run.py
        $ ./run.py

The whole process should take around 8 hours, depending on the machine that the script is running on. You will be able to keep track by looking at `results.log`.

The project report is `report.md`.

The structure of the pipeline is as follows:

        DonorsChoose-machine-learning
        ├── data: the folder containing the dataset for the project
        ├── evaluations: the folder containing the evaluation of different models created for temporally-split datasets
        ├── images: the folder containing precision-recall curves for the best model for each temporally-split test set
        ├── config.py: helper functions for logging and loading configurations
        ├── config.yaml: configuration file for the pipeline
        ├── environment.yml: Conda env file
        ├── extract_data.py: helper functions for loading the dataset
        ├── log_config.conf: configuration file for the log
        ├── model.py: helper functions for machine learning modeling
        ├── preprocessing.py: helper functions for preprocessing/cleaning the data
        ├── report.md: PROJECT REPORT
        ├── results.log: log file output by running run.py 
        ├── run.py: python script file for running the entire pipeline
        └── viz.py: helper functions for visualization

