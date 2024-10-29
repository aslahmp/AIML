import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'C:\Users\anuar\OneDrive\Desktop\logisticregression_data.xlsx'
df = pd.read_excel(file_path)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("Dataset preview:")
print(df)

df['Not Cured'] = df['Total Patients']-df['Cured']
df['Probability of Cured'] = df['Cured'] / df['Total Patients']
df['Probability of Not Cured'] = df['Not Cured'] / df['Total Patients']
df['P_cured/P_not_cured'] = df['Probability of Cured'] / df['Probability of Not Cured']
df['logit'] = np.log(df['P_cured/P_not_cured'])

print("Updated Dataset:")
print(df)

X = []
Y = []

for i in range(len(df)):
   X.append(df["Medication Dosage"].iloc[i])
   Y.append(df["logit"].iloc[i])

X = np.array(X)
Y = np.array(Y)

print("X (Medication Dosage):", X)
print("Y (logit):", Y)

mean_x = np.mean(X)
mean_y = np.mean(Y)

b = (np.sum(X * Y) - len(X) * mean_x * mean_y) / (np.sum(X**2) - len(X) * mean_x**2)
a = mean_y - b * mean_x

Y_pred = a + b * X

print(f"a={a}")
print(f"b={b}")

d = df["Medication Dosage"]
d_1 = a + b * d
d_2 = np.exp(d_1)
d_3 = d_2/(1+d_2)

print(d_3)

c_1 =[]
c_2 =[]
for value in d_3:
   if value > 0.5:
       c_1.append(value)
   else:
       c_2.append(value)

print(c_1)
print(c_2)

degree = 3
coefficients = np.polyfit(X, Y_pred, degree)
polynomial = np.poly1d(coefficients)

x_fine = np.linspace(min(X), max(X), 100)
y_fitted = polynomial(x_fine)

probability_fitted = np.exp(y_fitted) / (1 + np.exp(y_fitted))

plt.scatter(X, d_3,label='Original Data', color='blue')
plt.plot(x_fine, probability_fitted, color='red', label='Fitted Curve')
plt.axhline(y=0.5, color='green', linestyle='--', label='Threshold y=0.5')
plt.xlabel('Medical dosage')
plt.ylabel('Cured Patients')
plt.title('Analysis')
plt.legend()
plt.xlim([0, max(X) + 10])  # Adjust x limits if needed
plt.ylim([0, 1])
plt.show()
