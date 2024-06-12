import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns #Seaborn is a Python data visualization library based on matplotlib.
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score


#classification
data = pd.read_csv('zomato.csv')
print('Table data ')
data.head()
data.shape
#print(data.describe().T)
#print(data)

# removing unwanted coulmns
zomato=data.drop(['restaurant name'],axis=1)

#no duplicate values
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)

zomato.isnull().sum()
zomato.dropna(how='any',inplace=True)
zomato.info()

#cost 
zomato['cost'] = zomato['avg cost (two people)'].astype(str)
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.'))
zomato['cost'] = zomato['cost'].astype(float)
#zomato.info()

#---------
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.table_booking.replace(('Yes','No'),(True, False),inplace=True)
zomato.cost.unique()

#encode
def Encode(zomato):
    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:
        zomato[column] = zomato[column].factorize()[0]
    return zomato

zomato_en = Encode(zomato.copy())

corr = zomato_en.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
zomato_en.columns
#print(zomato_en.columns)
plt.show()

#Defining the independent variables and dependent variables
x = zomato_en.iloc[:,[1,2,3,4,5]]
y = zomato_en['rate']

#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
x_train.head()
y_train.head()


#Linear Regression model
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
#print(y_pred)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
#print(r2_score)

#sns.countplot(zomato['online_order'])
sns.countplot(data=zomato, x='online_order', order=[True, False])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or Not')
plt.show()

sns.countplot(data=zomato, x='table booking',order=[True,False])
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants allowing table booking or not')
plt.show()

plt.rcParams['figure.figsize'] = (13, 9)
Y = pd.crosstab(zomato['rate'], zomato['table booking'])
Y.div(Y.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['hotpink','black'])
plt.title('table booking vs rate', fontweight = 30, fontsize = 20)
plt.ylabel('table booking')
plt.legend(loc="upper right")
plt.show()



# Create a pie chart for the 'type' column
data=data.head(20)           #first 20 rows
type_counts = data['restaurant_type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Restaurant Types')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Create a pie chart for the 'cuisine_type' column
cuisine_counts = data['cuisines_type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(cuisine_counts, labels=cuisine_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Cuisine Types')
plt.axis('equal')
plt.show()


#graph between local address and the restaurant name

data = {
     'local address': ['Bellandur', 'Marathahalli', 'Basavanagudi', 'HSR', 'Frazer Town'],
    'name': ['#FeelTheROLL', "'@ Biryani Central", '5 Star Food', '7 Plates', '99Foods']
}

df = pd.DataFrame(data)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(df['local address'], df.index)
plt.xlabel('Name')
plt.ylabel('Local Address')
plt.title('Local Address vs. Name')
plt.show()


#import pandas as pd
#import matplotlib.pyplot as plt

# Sample data (you would use your DataFrame 'df' here)
data = {
    'name': ['5 Star Food', '7 Plates', '99Foods', 'A1 Biriyani Point', 'Eggzotic'],
    'rate': [4.2, 3.9, 4.5, 3.7, 4.1]
}

data1 = pd.DataFrame(data)

# Limit the DataFrame to the first 30 rows
data1 = data1.head(30)

# Create a bar plot
plt.figure(figsize=(12, 6))
plt.barh(data1['name'], data1['rate'], color='hotpink')
plt.xlabel('Ratings')
plt.ylabel('Restaurant Name')
plt.title('Restaurant Ratings of First 30 Rows')
plt.gca().invert_yaxis()  # Invert the y-axis to display names from top to bottom
plt.xlim(0, 5)  # x range from 0 to 5 
plt.show()


#graph
loc_plt=pd.crosstab(zomato['rate'],zomato['area'])
loc_plt.plot(kind='bar',stacked=True);
plt.title('Location - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Location',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');
plt.legend().remove();
plt.show()

#graph
df1=pd.read_csv('zomato_dataset.csv')
highRated_restaurants = df1[(df1['Dining_Rating'] > 4.0) & (df1['Delivery_Rating'] > 4.0)]
highRated_restaurants.head()
# For restaurant_name w.r.t. Dining Rating
restaurant_ratings = highRated_restaurants.groupby('Restaurant_Name')['Dining_Rating'].max()
# top 5 restaurants based on  Dining ratings
top_restaurants = restaurant_ratings.nlargest(5)

# Creating a bar graph
plt.figure(figsize=(6, 5))
top_restaurants.plot(kind='bar', color='teal')
plt.title('Top 5 Restaurants by Dining Rating')
plt.xlabel('Restaurant Name')
plt.ylabel('Dining Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


popular_items = highRated_restaurants.groupby('Item_Name')["Votes"].max()
popular_items = popular_items.nlargest(5)
plt.figure(figsize=(6, 6))
sns.set_palette('YlGnBu')  
plt.pie(popular_items, labels=popular_items.index ,autopct='%.1f%%', startangle=140)
plt.title('Top 5 Item in the Dataset')
plt.show()

#one more----------------------
cuisine_popularity = highRated_restaurants.groupby('Cuisine ')[['Dining_Votes', 'Delivery_Votes']].sum()

# Calculate the total votes for each cuisine by summing dining and delivery votes
cuisine_popularity['Total Votes'] = cuisine_popularity['Dining_Votes'] + cuisine_popularity['Delivery_Votes']

# Sort the cuisines by total votes in descending order
sorted_cuisine_popularity = cuisine_popularity.nlargest(5, 'Total Votes')

# Create a bar plot to visualize cuisine popularity and total votes
plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_cuisine_popularity.index, y='Total Votes', data=sorted_cuisine_popularity, palette='viridis')
plt.xlabel('Cuisine')
plt.ylabel('Total Votes')
plt.title(' Top 5 cuisines with the highest number of votes.')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#CONFUSION METRIX

# Load your Zomato restaurant dataset into a DataFrame (replace 'zomato.csv' with your dataset file)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load your Zomato dataset into a DataFrame
#data = pd.read_csv('zomato_dataset.csv')

# Correct data types for columns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create a sample DataFrame with online_order, table_booking, and rate columns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Create a sample DataFrame with online_order, table_booking, and rate columns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Create a sample DataFrame with online_order, table_booking, and rate columns
data = {
    'online_order': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'table booking': ['Yes', 'No', 'No', 'Yes', 'No'],
    'rate': [4.5, 3.8, 4.0, 4.2, 3.5]
}

df = pd.DataFrame(data)

# Define a rating threshold to classify as "High Rating" or "Low Rating"
threshold = 3.0
df['rating_class'] = df['rate'].apply(lambda x: 'High Rating' if x > threshold else 'Low Rating')

# Split the data into features (X) and target (y)
X = df[['online_order', 'table booking']]
y = df['rating_class']

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['online_order', 'table booking'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=40)

# Train a simple classifier (e.g., Random Forest)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Create a confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Create a Seaborn heatmap for the confusion matrix
plt.figure(figsize=(10, 5))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Rating', 'High Rating'], yticklabels=['Low Rating', 'High Rating'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# You can also print a classification report for more details
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_rep)


#K-Means ALGO
