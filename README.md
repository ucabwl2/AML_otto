# AML_otto
It is a Kaggle competition, working as a team to build a predictive model based on a 3-layer learning architecture to distinguish product categories. 
I have built this model with data pre-processing, integrating different classification models, utilizing Python libraries such as Sci-kit Learn, NumPy and Pandas. 
We ranked at top 20 at Kaggle finally.

o 1st level: there are about 36 models (KNN, Xgboost, Lasagne NN, T-sne reduction) that
are used for the predictions as meta feature for the 2nd level.

o 2nd level: 4 models are trained using 36 meta features from the 1st level. A cross-validate
is trained to choose the best model, tune hyperparameters and find optimum weights to
average 3rd level.

o 3rd level: Composed by a weighted mean of 2nd level predictions
