# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Step 2: Load and explore the data
data = pd.read_csv('data/housing.csv')
print(data.head())

# Step 3: Data Preprocessing
# Handle missing values
data.fillna(data.mean(), inplace=True)

# Scale numerical features
scaler = StandardScaler()
data[['Price', 'Size']] = scaler.fit_transform(data[['Price', 'Size']])

# Step 4: Split data into train and test sets
X = data[['Size']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
predictions = model.predict(X_test)
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, predictions, color='red')
plt.show()

