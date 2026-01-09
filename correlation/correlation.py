import pandas as pd

df = pd.read_excel("points.xlsx")
print(df)

print(df.corr())


import matplotlib.pyplot as plt

plt.scatter(df["x"], df["y"])
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Scatter Plot of X and Y Values")
plt.grid(True)

plt.savefig("correlation_scatter.png")
plt.show()
