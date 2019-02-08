# k-NN Classifier

We implement the brute-force k-Nearest Neighbors Classifier and use the implementation 
to study the wine-white dataset. All the required helper functions and performance metrics 
are also implemented. The implementations are cross checked with scikit learn.

* knn_classifier.py: KNeighborsClassifier class. Methods: fit, predict, predict_proba

* helper_functions.py: train_test_split, scale_normal - feature normalization.

* metrics.py: Performance metrics functions - accuracy_score, precision_score, recall, f1-score, roc_curve, precision_recall etc.
 
* distances.py: manhattan_distance, euclidean_distance etc.

* cross_validation.py: kfold_cross_validation.

* testing_knn_model.py: Using the custom implementation, we explore the wine-white dataset. 
