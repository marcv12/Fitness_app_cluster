# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv("intensityVSdiversity.csv")
df.head()

# plt.figure(0)
# fig = plt.scatter(df["Number_of_programs_taken"], df["Intensity_per_workout"])
# plt.show()

numeric_col = ['Number_of_programs_taken', 'Intensity_per_workout']

plt.figure()
df.boxplot(numeric_col)
plt.show()


for x in ['Number_of_programs_taken']:
    q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    df.loc[df[x] < min, x] = np.nan
    df.loc[df[x] > max, x] = np.nan


for y in ['Intensity_per_workout']:
    q75, q25 = np.percentile(df.loc[:, y], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    df.loc[df[y] < min, y] = np.nan
    df.loc[df[y] > max, y] = np.nan


print(df.isnull().sum())
df = df.dropna(axis = 0)

print(df.isnull().sum())


plt.figure(2)
df.boxplot(numeric_col)
plt.show()


plt.figure(3)
fig = plt.scatter(df["Intensity_per_workout"], df["Number_of_programs_taken"])
plt.show()




print(df[['Intensity_per_workout']].ndim)
print(df[['Number_of_programs_taken']].ndim)

scaler = MinMaxScaler()
scaler.fit_transform(df[['Number_of_programs_taken']])
df[['Number_of_programs_taken']] = scaler.fit_transform(df[['Number_of_programs_taken']])
print(df["Number_of_programs_taken"])
print(df.head())

scaler.fit_transform(df[['Intensity_per_workout']])
df[['Intensity_per_workout']] = scaler.fit_transform(df[['Intensity_per_workout']])
print(df["Intensity_per_workout"])





km = KMeans(n_clusters=3)
y_pred = km.fit_predict(df[['Intensity_per_workout', 'Number_of_programs_taken']])
print(y_pred)


df["cluster"] = y_pred
print(df.head())


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1["Number_of_programs_taken"], df1['Intensity_per_workout'], color='green')
plt.scatter(df2["Number_of_programs_taken"], df2['Intensity_per_workout'], color='red')
plt.scatter(df3["Number_of_programs_taken"], df3['Intensity_per_workout'], color='black')

plt.xlabel('Number_of_programs_taken')
plt.ylabel('Intensity_per_workout')

print(km.cluster_centers_)

k_rng = range(1, 18)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Intensity_per_workout', 'Number_of_programs_taken']])
    sse.append(km.inertia_)

print(sse)


plt.figure(6)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()


# df.to_csv("data_with_clusters.csv", index = False)
# print(df)

data = pd.read_csv("data_with_clusters.csv")
print(data)