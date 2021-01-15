import numpy as np
from sklearn import linear_model
from sklearn import metrics



def Mul_features( X_train , y_train , X_test , y_test):
    liner_mul_features = linear_model.LinearRegression()
    liner_mul_features.fit(X_train, y_train)
    prediction = liner_mul_features.predict(X_test)

    print('Co-efficient of linear regression', liner_mul_features.coef_)
    print('Intercept of linear regression model', liner_mul_features.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(np.asarray(y_test), prediction)))
    print(liner_mul_features.score(X_test,y_test))
