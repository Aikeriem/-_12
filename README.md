import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Загрузка данных из CSV-файла
file_path = "Fashion_Retail_Sales (1).csv"  # укажите свой путь к файлу
data = pd.read_csv(file_path)

# Вывод основной информации о данных
print(data.info())

# Вычисление перцентилей для Purchase Amount (USD)
percentiles = [25, 50, 75, 90, 95, 99]
purchase_amount_percentiles = np.percentile(data["Purchase Amount (USD)"], percentiles)

# Вывод результатов
print("\nPurchase Amount (USD) Percentiles:")
for p, value in zip(percentiles, purchase_amount_percentiles):
    print(f"{p}th percentile: {value}")

# Визуализация графика перцентилей
plt.figure(figsize=(8, 6))
plt.boxplot(data["Purchase Amount (USD)"], vert=False, showfliers=False)
plt.scatter(purchase_amount_percentiles, [1] * len(purchase_amount_percentiles), color='red', marker='o', label='Percentiles')
plt.title('Purchase Amount (USD) Percentiles')
plt.xlabel('Purchase Amount (USD)')
plt.yticks([])
plt.legend()
plt.show()

import pandas as pd


import numpy as np

# Генерация случайных данных для примера
data = np.random.randn(1000)  # случайные данные, распределенные по нормальному закону

# Вычисление стандартного отклонения
std_dev = np.std(data)

print(f"Standard Deviation: {std_dev}")


import pandas as pd

# Загрузка данных из CSV-файла
file_path = "Fashion_Retail_Sales (1).csv"  # укажите свой путь к файлу
df = pd.read_csv(file_path)

# Вычисление дисперсии для столбца "Purchase Amount (USD)"
variance_purchase_amount = df["Purchase Amount (USD)"].var()

print(f"Variance of Purchase Amount: {variance_purchase_amount}")

import pandas as pd

# Загрузка данных из CSV-файла
file_path = "Fashion_Retail_Sales (1).csv"  # укажите свой путь к файлу
df = pd.read_csv(file_path)

# Вычисление корреляции между столбцами "Purchase Amount (USD)" и "Review Rating"
correlation_coefficient = df["Purchase Amount (USD)"].corr(df["Review Rating"])

print(f"Correlation Coefficient: {correlation_coefficient}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных из CSV-файла
file_path = "Fashion_Retail_Sales (1).csv"  # укажите свой путь к файлу
df = pd.read_csv(file_path)

# Убедимся, что все столбцы, которые не числовые, исключены
numerical_df = df.select_dtypes(include=['number'])

# Построение матрицы корреляции
correlation_matrix = numerical_df.corr()

# Визуализация матрицы корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Создание случайных данных
np.random.seed(42)
X = np.random.rand(100)  # Некоторая переменная X
Y = 2 * X + 1 + 0.1 * np.random.randn(100)  # Переменная Y, зависящая от X

# Вычисление корреляции
correlation_coefficient, p_value = pearsonr(X, Y)

# Визуализация данных и регрессионной прямой
plt.scatter(X, Y, label=f'Correlation: {correlation_coefficient:.2f}')
plt.plot(X, 2 * X + 1, color='red', label='True Relationship: Y = 2X + 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Correlation vs. Causality Example')
plt.show()

# Вывод результата корреляции
print(f"Correlation Coefficient: {correlation_coefficient}")
print(f"P-value: {p_value}")
