import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

def Poly(  X_train , y_train , X_test , y_test ):
    poly_features = PolynomialFeatures(degree=2)

    X_train_poly = poly_features.fit_transform(X_train)

    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    y_train_predicted = poly_model.predict(X_train_poly)

    prediction = poly_model.predict(poly_features.fit_transform(X_test))

    print('Co-efficient of linear regression', poly_model.coef_)
    print('Intercept of linear regression model', poly_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(y_test, prediction)) )
    print("score", r2_score(y_test , prediction))
    #print(poly_model.score(X_test , y_test))
    #Accuracy = 1 - np.mean(abs((prediction - np.mean(prediction)) / np.mean(prediction)))
    #print(Accuracy)
