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


| Train Set               | Test Set                |
|-------------------------|-------------------------|
| 2012/01/01 - 2012/05/01 | 2012/07/01 - 2012/12/31 |
| 2012/01/01 - 2012/11/01 | 2013/01/01 - 2012/06/30 |
| 2012/01/01 - 2013/05/01 | 2013/07/01 - 2013/12/31 |


### Models

Seven classifiers (i.e. Logistic Regression, K-Nearest Neighbor, Decision Trees,
