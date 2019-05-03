# Machine Learning with DonorsChoose

## Getting Started

I used conda as the environment manager.

### Prerequisites

1. Clone the repository.

        $ git clone https://github.com/capp-machine-learning/DonorsChoose-machine-learning.git
        $ cd DonorsChoose-machine-learning

1. A yml file of the environment is available in environment.yml.

        $ conda env create --file=environment.yml
        $ conda activate donorsml
        
### Files

1. For the general workflow of the code and pipeline, look at DonorsChoose-Machine-Learning.ipynb

2. The pipeline is divided into 6 files:
            ├── config.py: load the config file
            ├── config.yaml: config file
            ├── model.py: train and evaluate models
            ├── preprocessing.py: preprocess data
            ├── pipeline.py: other helper functions
            └── extract_data.py: load the data


