import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../dataset/laptop_prices.csv')

numerical_features = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']

numerical_data = data[numerical_features].copy()
numerical_data.fillna(numerical_data.mean(), inplace=True)
data[numerical_features] = numerical_data

scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

data = pd.get_dummies(data, columns=['Brand'], drop_first=True)

corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.show()

threshold = 0.8
correlated_features = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            if colname != 'Price':
                correlated_features.add(colname)

data.drop(list(correlated_features), axis=1, inplace=True)

for column in data.columns:
    if data[column].dtype == 'float64' or data[column].dtype == 'int64':
        if len(np.unique(data[column])) > 10:
            data[column] = pd.cut(data[column], bins=5, labels=False)

X = data.drop('Price', axis=1)
y = data['Price']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Получение важности признаков
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False, inplace=True)

print("Важность признаков (Gain Ratio):")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")

