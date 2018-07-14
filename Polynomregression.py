import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veri=pd.read_csv('verim.csv')
x=veri.iloc[:11,0:1]
y=veri.iloc[:11,1:2]
X=x.values
Y=y.values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=6)
x_poly=poly_reg.fit_transform(X)

poly_reg=LinearRegression()
poly_reg.fit(x_poly,y)
plt.scatter(x,y)
plt.plot(x,poly_reg.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

print(poly_reg.predict(poly_reg.transform(11)))
print(poly_reg.predict(poly_reg.transform(1)))
print(x_poly)
