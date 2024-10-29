import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'C:\Users\anuar\OneDrive\Desktop\linearregression_data.xlsx'
df = pd.read_excel(file_path)

print("Dataset preview:")
print(df)

X = []
Y = []

for i in range(len(df)):
    X.append(df["YearsExperience"].iloc[i])
    Y.append(df["Salary"].iloc[i])

X = np.array(X)
Y = np.array(Y)

print("X (YearsExperience):", X)
print("Y (Salary):", Y)

mean_x = np.mean(X)
mean_y = np.mean(Y)

b = (np.sum(X * Y) - len(X) * mean_x * mean_y) / (np.sum(X**2) - len(X) * mean_x**2)
a = mean_y - b * mean_x

print("\nEquation of line is in the form Y=MX+C")
print(f"M= {b}")
print(f"C= {a}")

Y_pred = a + b * X

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Least Squares Fitting')
plt.show()
