# COMP20008_Ass2
Assignment 2 for Elements of Data Processing

# please git pull before making any changes.
# it may also be beneficial to make any changes on a different branch to the main. 

To run preprocessing.py:
    python main.py preprocessing
    This section applies pre-prossessing to each "filtered" CSV. 

To run Correlation section:
    Code is mostly self-contained, run preprocessing first, then in terminal
    python ./Correlation/all_correlation.py
    This will generate all relevant plots in ./Correlation/Figures and print out some results from regression model (only r coefficient relevant for report)
    Note plots will differ from report due to inherent randomness of method, but general relative strengths are consistent

To run Supervised Machine Learning Section:
    python machine_learning/machine_learning_models.py machine_learning/model_eval_helpers.py
    This section evaluates and trains the two models, evaluation statistics and hyperparameter selection is printed in the evaluations.txt
    Additionally, plots of each confusion matrix is also saved to the machine_learning directory

To run Clustering Section:
    The following commands need to be run in the Ryan Clustering directory
    python ageHazardPerception.py
    python dayTimeCluster.py
    python dayTimeClusterNoWeight.py
    python timeClusteringMerged.py
    This section produces the clustered scatter-plots and elbow graphs used in the report.
        