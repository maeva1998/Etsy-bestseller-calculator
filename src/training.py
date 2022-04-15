"""
Creating different ready to use models
"""
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# (weight?) deep NNET (look for regression with NNET)

#Support Vector Regression
def SVM(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    regressor = SVR(kernel='rbf')
    regressor.fit(X, y)


#Lasso Regression

#Random Forest
