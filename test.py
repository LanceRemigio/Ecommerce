import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

sns.set_style(style = 'darkgrid')

# Read in csv files and assign to variable named "customers"
customers = pd.read_csv('Ecommerce Customers')

# Check the had of customers, and check out its info() and describe() methods.

print(
        customers.head()
        )

print(
        customers.info()
        )

print(
        customers.describe()
        )


print(
        customers.columns
        )


# Exploratory Data Analysis

# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent Columns. Does the correlation make sense?

sns.jointplot(
        data = customers, 
        x  = customers['Time on Website'],
        y = customers['Yearly Amount Spent']
        )

# plt.savefig('./figures/jointplot.png')

# do the same but this time with time = 'Time on App' instead.

sns.jointplot(
        data = customers,
        x = customers['Time on App'],
        y = customers['Yearly Amount Spent']
        )

# plt.savefig('./figures/jointplotTimeOnApp.png')

# Plot the relationship between 'Time on App' and 'Length of Membership'

sns.jointplot(
        data = customers,
        kind = 'hex' ,
        x = customers['Time on App'],
        y = customers['Length of Membership']
        )

# plt.savefig('./figures/hexplot.png')

sns.pairplot(
        data = customers
        )

# plt.savefig('./figures/pairplot.png')

sns.lmplot(
        data = customers, 
        x = 'Length of Membership',
        y = 'Yearly Amount Spent'
        )

# plt.savefig('./figures/lmplot.png')


# Traning and Testing Data
X = customers[[
    'Avg. Session Length', 
    'Time on App', 
    'Time on Website', 
    'Length of Membership', 
     ]]

y = customers['Yearly Amount Spent']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)

# # Training the Model

lm = LinearRegression()

lm.fit(X_train, y_train) # does not need to be assigned to a variable

print(lm.coef_) # prints out coefficients of model

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions) # creates scatter plot of real test values vs predicted values
plt.xlabel('Y test')
plt.ylabel('Predicted Y')
plt.savefig('./figures/scatter.png')

## Evaluating the Model

coeff_customers = pd.DataFrame(lm.coef_ , X.columns, columns = ['Coefficient'])

print(coeff_customers)

evalResults = [
        'MAE: ' + str(metrics.mean_absolute_error(y_test, predictions)),
        'MSE: ' + str(metrics.mean_squared_error(y_test, predictions)),
        'RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
        ] 

for item in evalResults:
    print(item, end = '\n')

sns.displot((y_test - predictions), kde = True, bins = 50) 
plt.savefig('./figures/histplot.png')
plt.show()
