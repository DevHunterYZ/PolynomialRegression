import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veri=pd.read_csv('QBOdata.csv')
x=veri.iloc[:25,0:1]
y=veri.iloc[:25,1:2]
X=x.values
Y=y.values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X,Y,color='red')

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(X)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y)
plt.title('Son 25 aylık QBO değişimi')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='green')
plt.show()

print(lin_reg2.predict(poly_reg.fit_transform(25)))
print(lin_reg2.predict(poly_reg.fit_transform(26)))
