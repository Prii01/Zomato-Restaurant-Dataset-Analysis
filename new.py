import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from   sklearn.impute import SimpleImputer

df=pd.read_csv('zomato.csv')
print(df)
df.drop(axis=1 , columns=['Unnamed: 0.1' , 'Unnamed: 0'] , inplace=True)
df.head(20)

df.shape
df.info()

# removing unwanted coulmns
df=df.drop(['restaurant name'],axis=1)

#removing the null values
df.isnull().sum()
df.info()

df["rate (out of 5)"].fillna(df["rate (out of 5)"].mean(),inplace=True)
sns.kdeplot(df["avg cost (two people)"])
plt.title('Avg cost(two people)')
plt.show()
print(df)

df["avg cost (two people)"].fillna(df["avg cost (two people)"].median(),inplace=True)
#check for null values
df.isnull().sum()

ratings = df["rate (out of 5)"]
avg_cost = df["avg cost (two people)"]
plt.subplot(1,2,1)
ratings.plot.box(title="Rating (out of 5)", xticks=[])

plt.subplot(1,2,2)
avg_cost.plot.box(title="Avg cost (two people)", xticks=[])
plt.show()

#starting values
df['area'].value_counts().head()

#ending values
df['area'].value_counts().tail()

df.sort_values(by='num of ratings',ascending=False).head()
df.sort_values(by='num of ratings',ascending=False).tail()

#1st graph
plt.hist(df['rate (out of 5)'],bins =10)
plt.xlabel('ratings out of 5')
plt.ylabel('Frequency')
plt.title('Frequency of ratings')
plt.show()

#Scatter plot
plt.scatter(df['avg cost (two people)'],df['num of ratings'])
plt.ylabel('num of rating')
plt.xlabel('average cost per two people')
plt.title('Variation of num of ratings based on avg cost')
plt.show()

#2nd Scatter plot
plt.scatter(df['rate (out of 5)'],df['avg cost (two people)'])
plt.xlabel('ratings out of 5')
plt.ylabel('average cost per two people')
plt.title('Variation of ratings based on avg cost')
plt.show()

#pie chart -> with the breakdown
data = df['restaurant type'].value_counts().head(10)
labels = data.index
explode_val = [0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08,0.08]
plt.pie(data, labels = labels, autopct = '%1.1f%%',explode=explode_val)
plt.title('Top 10 restaurant types')
plt.show()

#2nd pie chart-> with break down
data = df['restaurant type'].value_counts().tail(5)
labels = data.index
exp_val = [0.06,0.06,0.06,0.06,0.06]
plt.pie(data, labels = labels, autopct = '%1.1f%%',explode=exp_val)
plt.title('5 restaurant types from which food is not ordered very frequently')
plt.show()

#1st bar graph of Online order and Offline order
df['restaurant type'].unique()
Top_Ten_Types = df['restaurant type'].value_counts()[:10]

plt.figure(figsize=(12,8))
sns.set_style('darkgrid')
sns.barplot(x=Top_Ten_Types.values, y=Top_Ten_Types.index)
plt.title('Top Ten Types' , size = 20)
plt.xlabel('Number' , size = 15)
plt.ylabel('Kind' , size = 15)
plt.show()

#Online_order and offline order
df['online_order'].value_counts()
df['table booking'].value_counts()
Boolean_features = ['online_order','table booking']
plt.figure(figsize=(10 , 5))
for inx , value in enumerate(Boolean_features):
    plt.subplot(1 , 2 , inx + 1)
    plt.pie(df[value].value_counts() , labels=df[value].value_counts().keys() ,
            autopct='%1.1f%%' ,explode=[0.005 , 0.1])
    plt.title(f'{Boolean_features[inx]} ')
plt.show()

#next thing
data=pd.read_csv('zomato_dataset.csv')
print(data)


# Assuming your DataFrame 'data' contains the mentioned columns

plt.figure(figsize=(10, 6))

# Create a new DataFrame containing only the columns you want to include in the correlation heatmap
selected_columns = data[['Dining_Rating', 'Delivery_Rating', 'Dining Votes', 'Delivery_Votes', 'Votes', 'Prices']]

# Calculate the correlation matrix for the selected columns
correlation_matrix = selected_columns.corr()

# Create the correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Correlation Heat Map')
plt.show()

#K-means cluster
from sklearn.cluster import KMeans

# select featur for clustering
x_cluster = data[['Dining Votes', 'Delivery_Votes', 'Votes', 'Prices']]

# fit KMeans model
kmeans_model = KMeans(n_clusters=7, random_state=42)
data['Cluster'] = kmeans_model.fit_predict(x_cluster)

# visualize the cluster
plt.figure(figsize=(10, 7))
sns.scatterplot(x = 'Dining Votes', y = 'Prices', hue='Cluster', data = data)
plt.xlabel('Dining Votes')
plt.ylabel('Prices')
plt.title('KMeans Clustering : Dining Votes vs. Prices')
plt.show()

#Confusion metrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame 'data' with 'online_booking' column containing 'yes' and 'no' values
predicted_labels = df['online_order']
true_labels = df['online_order']  # Using the same column for true labels

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=['Yes', 'No'])

# Create a heatmap for the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted Yes', 'Predicted No'], yticklabels=['Actual Yes', 'Actual No'])
print(conf_matrix)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Online Booking')
plt.show()

#pie chart
location_counts = data['City'].value_counts().head(10)
colors = ['#FFA07A', '#FFB38D', '#FFCBA4', '#FFE1BD', '#FFFFCC', '#D4EFDF', '#AED6F1', '#F8C471', '#D2B4DE', '#F9E79F']

plt.figure(figsize=(8, 6))
plt.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%', startangle=90,colors = colors)

plt.title('Locations with Highest Number of Restaurants', fontsize=14, weight = 'bold')

plt.tight_layout()
plt.show()
