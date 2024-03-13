import numpy as np
import matplotlib.pyplot as mp
from sklearn.linear_model import LinearRegression
heights = np.array([160, 165, 170, 175, 180, 185, 190]).reshape(-1, 1)
weights = np.array([55, 60, 63, 70, 75, 77, 80]).reshape(-1, 1)
model = LinearRegression()
model.fit(heights, weights)
new_height = np.array([[170]])
predicted_weight = model.predict(new_height)
print(f"Predicted weight for a height of {new_height[0][0]} cm: {predicted_weight[0][0]} kg")
mp.scatter(heights, weights, color='blue', label='Data Points')
mp.plot(heights, model.predict(heights), color='black', linewidth=3, label='Regression Line')
mp.scatter(new_height, predicted_weight, color='red', marker='*', s=200, label='Predicted Weight')
mp.xlabel('Height (cm)')
mp.ylabel('Weight (kg)')
mp.title('Linear Regression: Height vs. Weight')
mp.legend()
mp.show()