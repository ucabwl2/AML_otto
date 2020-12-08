# AML_otto
It is a Kaggle competition, working as a team to build a predictive model based on a 3-layer learning architecture to distinguish product categories. 
I have built this model with data pre-processing, integrating different classification models, utilizing Python libraries such as Sci-kit Learn, NumPy and Pandas. 
We ranked at top 20 at Kaggle finally.

o 1st level: there are about 36 models 
-Model 1: RandomForest(R). Dataset: X

-Model 2: Logistic Regression(scikit). Dataset: Log(X+1)

-Model 3: Extra Trees Classifier(scikit). Dataset: Log(X+1) (but could be raw)

-Model 4: KNeighborsClassifier(scikit). Dataset: Scale( Log(X+1) )

-Model 5: libfm. Dataset: Sparse(X). Each feature value is a unique level.

-Model 6: H2O NN. Bag of 10 runs. Dataset: sqrt( X + 3/8)

-Model 7: Multinomial Naive Bayes(scikit). Dataset: Log(X+1)

-Model 8: Lasagne NN(CPU). Bag of 2 NN runs. First with Dataset Scale( Log(X+1) ) and second with Dataset Scale( X )

-Model 9: Lasagne NN(CPU). Bag of 6 runs. Dataset: Scale( Log(X+1) )

-Model 10: T-sne. Dimension reduction to 3 dimensions. Also stacked 2 kmeans features using the T-sne 3 dimensions. Dataset: Log(X+1)

-Model 11: Sofia(R). Dataset: one against all with learner_type="logreg-pegasos" and loop_type="balanced-stochastic". Dataset: Scale(X)

-Model 12: Sofia(R). Trainned one against all with learner_type="logreg-pegasos" and loop_type="balanced-stochastic". Dataset: Scale(X, T-sne Dimension, some 3 level 
interactions between 13 most important features based in randomForest importance )

-Model 13: Sofia(R). Trainned one against all with learner_type="logreg-pegasos" and loop_type="combined-roc". Dataset: Log(1+X, T-sne Dimension, some 3 level interactions between 13 most important features based in randomForest importance )

-Model 14: Xgboost(R). Trainned one against all. Dataset: (X, feature sum(zeros) by row ). Replaced zeros with NA.

-Model 15: Xgboost(R). Trainned Multiclass Soft-Prob. Dataset: (X, 7 Kmeans features with different number of clusters, rowSums(X==0), rowSums(Scale(X)>0.5), rowSums(Scale(X)< -0.5) )

-Model 16: Xgboost(R). Trainned Multiclass Soft-Prob. Dataset: (X, T-sne features, Some Kmeans clusters of X)

-Model 17: Xgboost(R): Trainned Multiclass Soft-Prob. Dataset: (X, T-sne features, Some Kmeans clusters of log(1+X) )

-Model 18: Xgboost(R): Trainned Multiclass Soft-Prob. Dataset: (X, T-sne features, Some Kmeans clusters of Scale(X) )

-Model 19: Lasagne NN(GPU). 2-Layer. Bag of 120 NN runs with different number of epochs.

-Model 20: Lasagne NN(GPU). 3-Layer. Bag of 120 NN runs with different number of epochs.

-Model 21: XGboost. Trained on raw features. Extremely bagged (30 times averaged).

-Model 22: KNN on features X + int(X == 0)

-Model 23: KNN on features X + int(X == 0) + log(X + 1)

-Model 24: KNN on raw with 2 neighbours

-Model 25: KNN on raw with 4 neighbours

-Model 26: KNN on raw with 8 neighbours

-Model 27: KNN on raw with 16 neighbours

-Model 28: KNN on raw with 32 neighbours

-Model 29: KNN on raw with 64 neighbours

-Model 30: KNN on raw with 128 neighbours

-Model 31: KNN on raw with 256 neighbours

-Model 32: KNN on raw with 512 neighbours

-Model 33: KNN on raw with 1024 neighbours

-Feature 1: Distances to nearest neighbours of each classes

-Feature 2: Sum of distances of 2 nearest neighbours of each classes

-Feature 3: Sum of distances of 4 nearest neighbours of each classes

-Feature 4: Distances to nearest neighbours of each classes in TFIDF space

-Feature 5: Distances to nearest neighbours of each classed in T-SNE space (3 dimensions)

-Feature 6: Clustering features of original dataset

-Feature 7: Number of non-zeros elements in each row

-Feature 8: X (That feature was used only in NN 2nd level training) 

o 2nd level: 4 models are trained using 36 meta features from the 1st level. A cross-validate
is trained to choose the best model, tune hyperparameters and find optimum weights to
average 3rd level.

o 3rd level: Composed by a weighted mean of 2nd level predictions
