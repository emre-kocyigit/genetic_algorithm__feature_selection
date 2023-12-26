# genetic_algorithm__feature_selection

Genetic Algorithm(GA) is applied to select the optimal subset of feature set to obtain the best classification score in ML project.

I used "Airline Passenger Satisfaction" dataset  which is available in the following link:
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download&select=train.csv

After preprocessing and undersampling of the majority class, I obtained 6526 rows and 22 features. Target is 'label' and it's binary. So, this is a binary classification problem.

I have two files:
- GeneticAlgorithm : Main file for the feature selection by GA
- MachineLearning : Secondary file to get fitness value by preparing data, building and trainig ML model, evaluating and returning the score.

Default GA parameters are defined at the beginning of the GeneticAlgorithm.py. You can customize them.

Output:
- A list which has the best chromosome of each generation --> to see the evolutionary progress
- Best chromosome which represents the optimal features as a part of feature selection
