import pandas as pd
import matplotlib.pyplot as plt

# Set up the output screen
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = [20, 12]

# Read dataset
trainData = pd.read_csv('./train.csv')

# With outliers
plt.scatter(trainData.GarageArea, trainData.SalePrice, color='red')
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.show()

# Delete the outlier value of GarageArea
outlier_drop = trainData[(trainData.GarageArea < 999) & (trainData.GarageArea > 111)]

# Display the scatter plot of GarageArea and SalePrice after deleting
plt.scatter(outlier_drop.GarageArea, outlier_drop.SalePrice, color='green')
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.show()