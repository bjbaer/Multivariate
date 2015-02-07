import pandas as pd
import numpy as np
import statsmodels.api as sm
import math

loansData = pd.read_csv('LoanStats3c.csv')
loansData.dropna(inplace=True)

#clean the data for interest rate
def g(x):
	x = x.rstrip('%')
	x = float(x)
	x = x / 100
loansData['int_rate'] = map(g, loansData['int_rate']) #overwrite old interest rate

intrate = loansData['int_rate']
aninc = loansData['annual_inc']

y = np.matrix(intrate).transpose() 
x = np.matrix(aninc).transpose()

X = sm.add_constant(x) #Big x at the start
model = sm.OLS(y,X) #Ordinary Least Squares Regression
f = model.fit()

print 'Coefficients: ', f.params[1:3]
print "Intercept: ", f.params[0]
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared
