# Project Report
## Si Young Byun


### Overview

__*DonorsChoose.org*__ is an online charity website that allows to help students in need through crowd funding projects. In this project, I looked at past project funding data to build and analyze machine learning models to predict whether a given project will be fully funded within 60 days of posting. A *good* predictive model would allow us to help projects that need help with fulfilling their goals. Specifically, the final goal of the project was to recommend a model that will best identify 5% of posted projects to intervene with.


### Data

The dataset spans from 01/01/2012 to 12/31/2013.

The dataset includes information on various aspects of the project:

- geographical location,
- primary and secondary focus of the project,
- resource type,
- poverty level,
- grade level,
- total price of the project,
- etc.

The outcome/dependent variable was created using `date_posted` and `datefullyfunded`. Specifically, if a project was funded within 60 days, the gap between `date_posted` and `datefullyfunded` would be less than 60 days.

I used the following features to train machine learning models for this project:

- Latitude/longitude of the school,
- The total price of the project,
- The number of students,
- The location of the project (Midwest, Northeast, South, West) -- *a feature created from `school_state`,*
- The area that the school is located in (Urban, Suburban, Rural),
- Whether the school is a charter/magnet school,
- The gender of the teacher -- *a feature created from `teacher_prefix`,*
- The primary focus area of the project,
- The resource type requested by the project,
- The grade level of students,
- Whether the project is eligible for double your impact match.

There were a number of missing values. For categorical features, missing values were imputed by the most frequent value. For continuous features, missing values were imputed by the mean or median depending on the skewness of the feature.

__Before__ preprocessing the data, the dataset was temporally split to take the temporal dimension into account. Specifically, three pairs of train/test datasets were created.


|   | Train Set               | Test Set                |
|---|-------------------------|-------------------------|
| 1 | 2012/01/01 - 2012/05/01 | 2012/07/01 - 2012/12/31 |
| 2 | 2012/01/01 - 2012/11/01 | 2013/01/01 - 2012/06/30 |
| 3 | 2012/01/01 - 2013/05/01 | 2013/07/01 - 2013/12/31 |


### Models

1. Seven classifiers (i.e. Logistic Regression, K-Nearest Neighbor, Decision Trees, SVM, Random Forests, Adaboosting, and Bagging) have been used to identify the best model for this project. Multiple combinations of parameters were tested for each classifier to find the parameters that produced the highest precision score for each classifier.

2. After tuning for parameters, different evaluation metrics (accuracy, precision at different levels, recall at different levels, F1, Area under Curve) were measured for each classifier.

3. Finally, the best model was selected by looking at the model with the highest `Precision_at_5%` score *as it is important to ensure that high proportion of positive identifications is actually correct in order to utilize the limited resources efficiently.* In an ideal world, all projects identified by my model would __NOT__ be funded within 60 days without interventions. Since I had three different train/test sets, I would have at most three different *best* models.


### Results and Recommendations

The following is the evaluation table of candidate models for the first train/test sets:

|model   |parameters                                                                           |Accuracy          |F1                |ROC_AUC           |Precision_at_1%   |Recall_at_1%        |Precision_at_2%   |Recall_at_2%        |Precision_at_5%   |Recall_at_5%        |Precision_at_10%  |Recall_at_10%      |Precision_at_20%  |Recall_at_20%      |Precision_at_30%  |Recall_at_30%     |Precision_at_50%  |Recall_at_50%     |
|--------|-------------------------------------------------------------------------------------|------------------|------------------|------------------|------------------|--------------------|------------------|--------------------|------------------|--------------------|------------------|-------------------|------------------|-------------------|------------------|------------------|------------------|------------------|
|LR      |{'C': 1, 'penalty': 'l2', 'random_state': 10, 'solver': 'liblinear'}                 |0.7329009074852305|0.8408080294752891|0.6420217385685597|0.9207317073170732|0.012376541945002255|0.9207317073170732|0.02475308389000451 |0.9128580134064594|0.061390926601368793|0.8924763935424916|0.12007704602270398|0.8728490939546216|0.23490840539322158|0.8533143843264643|0.3444940781115528|0.8198428649735063|0.5516577189459448|
|SVM     |{'C': 1, 'dual': False, 'loss': 'squared_hinge', 'penalty': 'l2', 'random_state': 10}|0.7415189719227724|0.8492040932347925|0.6282060592077756|0.9329268292682927|0.012540469652883078|0.9176829268292683|0.024671120036064097|0.8982327848872639|0.060407360354083846|0.8824246116356991|0.11872464243268718|0.8617329069590376|0.23191672472439653|0.8434676682570298|0.3405188311954428|0.8125342590900786|0.5467398877095201|
|RF      |{'max_features': 20, 'random_state': 10}                                             |0.6974846214751202|0.808001546192501 |0.6138535782927267|0.8932926829268293|0.0120077046022704  |0.8810975609756098|0.02368755378877915 |0.8756855575868373|0.05889102905618622 |0.8778556198598843|0.11810991352813409|0.8541190802497335|0.22986762837588623|0.8367678408283423|0.3378140240154092|0.8064437541872221|0.5426416950124995|
|BG      |{'max_samples': 0.1, 'n_estimators': 2, 'random_state': 10}                          |0.6967233083622633|0.8124588064704441|0.5491122905139003|0.75              |0.01008155403467071 |0.7439024390243902|0.019999180361460596|0.7629494210847044|0.05130937256669808 |0.7706366128540969|0.10368427523462154|0.7720420283234354|0.20777836973894512|0.7782966196325246|0.3142084340805705|0.7758694195748828|0.522068767673456 |
|KNN     |{'metric': 'manhattan', 'n_neighbors': 2, 'weights': 'uniform'}                      |0.7049759425056337|0.8192334962868978|0.5443757764956743|0.774390243902439 |0.01040940945043236 |0.7591463414634146|0.020408999631162657|0.7568555758683729|0.050899553296996025|0.7590618336886994|0.1021269620097537 |0.7627531597380843|0.20527847219376255|0.7699725916150645|0.3108479160690136|0.7716060661428833|0.5192000327855416|
|DT      |{'criterion': 'entropy', 'max_depth': 50, 'max_features': 30, 'random_state': 10}    |0.6286619160728424|0.7413676083821158|0.5458293331185453|0.7378048780487805|0.009917626326789886|0.7484756097560976|0.020122126142371213|0.7525898842169408|0.05061267980820458 |0.7544928419128846|0.1015122331052006 |0.7633622658748287|0.20544239990164337|0.766013602679931 |0.3092496209171755|0.7702661550642548|0.518298430392197 |
|Baseline|{'strategy': 'uniform'}                                                              |0.7430720506730008|0.8526004996593232|0.5               |0.75              |0.01008155403467071 |0.7362804878048781|0.019794270726609567|0.7282145033516149|0.048973402729396334|0.7343892780992994|0.098807425925167  |0.7428049337597076|0.19990983976066554|0.7411430311643488|0.299209048809475 |0.7400572507460869|0.4979713946149748|

The table shows that all models perform better than the baseline classifier in terms of `Precision_at_5%` with the logistic regression classifier performing the best. The below shows the precision-recall curve of this logistic regression model.

![prc1](./images/2012-07-01_0_prc.png)
