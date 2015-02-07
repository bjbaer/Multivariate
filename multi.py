import numpy as np
import pandas as pd
import statsmodels.api as sm

loansData = pd.read_csv('LoanStats3c.csv')
loansData.dropna(inplace=True)


g = lambda x: float(x.replace("%", ""))/100 #create lambda function to remove % and convert to float
loansData['int_rate'] = map(g, loansData['int_rate']) #overwrite old interest rate

intrate = loansData['int_rate'] #create new variable based on cleaned interest rate
annual_inc = loansData['annual_inc']

#transpose the variables into columns
y = np.matrix(intrate).transpose() 
x1 = np.matrix(annual_inc).transpose()

#create the linear model
X = sm.add_constant(x1) #Big x at the start
model = sm.OLS(y,X) #Ordinary Least Squares Regression
f = model.fit()

print 'Coefficients: ', f.params[1:3]
print "Intercept: ", f.params[0]
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared

#now add the variable home ownership. Need to turn this into dummy variables 
dummy_home = pd.get_dummies(loansData['home_ownership'], prefix = 'home')
mort_home = dummy_home['home_MORTGAGE']
own_home = dummy_home['home_OWN']
rent_home = dummy_home['home_RENT']

x2 = np.matrix(mort_home).transpose()
x3 = np.matrix(own_home).transpose()
x4 = np.matrix(rent_home).transpose()

#create the linear model
x = np.column_stack([x1, x2, x3, x4])
X = sm.add_constant(x) #Big x at the start
model = sm.OLS(y,X) #Ordinary Least Squares Regression
f = model.fit()

print 'Coefficients: ', f.params[1:]
print "Intercept: ", f.params[0]
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared
