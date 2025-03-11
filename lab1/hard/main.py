import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from lab1.hard.my_math import calculate_correlation_matrix, select_features_with_gain_ratio

data = pd.read_csv('../dataset/laptop_prices.csv')

numerical_features = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']

numerical_data = data[numerical_features].copy()
numerical_data.fillna(numerical_data.mean(), inplace=True)
data[numerical_features] = numerical_data

scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

data = pd.get_dummies(data, columns=['Brand'], drop_first=True)

corr_matrix, numeric_columns = calculate_correlation_matrix(data)
corr_df = pd.DataFrame(corr_matrix, columns=numeric_columns, index=numeric_columns)
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
plt.show()

threshold = 0.8
correlated_features = set()

for i in range(len(corr_df.columns)):
    for j in range(i):
        if abs(corr_df.iloc[i, j]) > threshold:
            colname = corr_df.columns[i]
            if colname != 'Price':
                correlated_features.add(colname)

data.drop(list(correlated_features), axis=1, inplace=True)

feature_importance = select_features_with_gain_ratio(data, 'Price')

print("Важность признаков (Gain Ratio):")
for importance in feature_importance:
    print(f"{importance[0]}: {importance[1]:.4f}")

