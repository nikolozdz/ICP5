import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('winequality-red.csv')

# Working with Numeric Features
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

corr = data.corr()
plt.figure(figsize=(20, 20));
sns.heatmap(corr, annot=True, cmap="YlGnBu")
plt.show();
print(corr['quality'].sort_values(ascending=False)[1:4], '\n')

print("Number of NaN values in Each column")
print(train.isnull().sum(axis=0))


#Transforming and engineering non-numeric features
train = data.apply(LabelEncoder().fit_transform)
# Build a multiple model
y = data['quality']
X = data.drop(['alcohol', 'sulphates', 'citric acid'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=.33)

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)


print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)

print ('RMSE is: \n', mean_squared_error(y_test, predictions))


actual_values = y_test
plt.scatter(predictions, actual_values,
            color='b')
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.title('Linear Regression Model')
plt.show()