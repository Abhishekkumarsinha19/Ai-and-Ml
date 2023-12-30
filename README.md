# Ai-and-Ml
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("C:\\Users\\hp\\Downloads\\train(2).csv")
df
df.head()
df.tail()
df.info()
df.columns
df.describe()
df.dtypes
name = df.isnull().sum().rename('num_of_missings').reset_index()
name.columns = ['Feature', 'num_of_missings']
name['percentage_of_missings'] = name['num_of_missings'] / len(df)
name = name.sort_values(by='percentage_of_missings', ascending=False)
name.style.background_gradient(cmap='Oranges')
plt.figure(figsize=(20,7))
plt.xticks(rotation=90)
sns.barplot(x='Feature', y='percentage_of_missings', palette='BrBG', data=name.sort_values(by='percentage_of_missings'))
numeric=['int32',"float64"]
df_numeric=df.select_dtypes(numeric)
df_numeric
df['Age'].mode()
sns.boxplot(df['Age'])
df['Age'].fillna(df['Age'].mode()[0],inplace=True)
df_obj=df.select_dtypes(include='object')
df_obj
for i in df_obj:
    if df_obj[i].isnull().sum().any():
        print(i)
df['Cabin'].mode()
df['Cabin'].fillna(df['Cabin'].mode()[0],inplace=True)
df['Embarked'].mode()
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
for i in df.col
umns:
    print(i,'<--------------->',df[i].isnull().sum()/df.shape[0]*100)
    df.drop(columns=['PassengerId','Ticket'],inplace=True)
    df['Cabin_name']=df['Cabin'].apply(lambda x:x[0])
    df.drop(columns=['Cabin'],inplace=True)
    df.duplicated().sum()
    for i in df.columns:
    print(i,'___________',df[i].unique())
    my_list = ['Survived', 'Pclass']
for i in my_list:
    df[i] = df[i].astype("object")
    df.select_dtypes(include='object')
    k=df['Survived'].value_counts()
k
my_list = ['Not_Survived', 'Survived']
my_list = ['Not_Survived', 'Survived']
plt.pie(k, labels=my_list, autopct="%0.0f%%", explode=[0.1, 0.2], colors=['olivedrab', 'gray'])
plt.title('Percentage of People who survived or Not')
plt.legend()
plt.show()
df['Pclass'].unique()
k1=df['Pclass'].value_counts()
k1
l = ["3", "1", "2"]

plt.pie(k1, labels=l, autopct="%0.0f%%", explode=[0.1]*len(l), colors=['olivedrab', 'gray', 'rosybrown'])
plt.title('Percentage of People of Pclass')
plt.legend()
plt.show()
df['Sex'].unique()
df['Sex'].value_counts()
gender_list = ['Male', 'Female']
gender_counts = [10, 15]  # Replace these counts with your actual data

plt.pie(gender_counts, labels=gender_list, autopct="%0.0f%%", explode=[0.1, 0.1], colors=['olivedrab', 'rosybrown'])
plt.title('Percentage of Sex')
plt.legend()
plt.show()
df['Embarked'].unique()
k3=df['Embarked'].value_counts()
k3
my_list = ['S', 'C', 'Q']
plt.pie(k3,labels=l,autopct="%0.0f%%",explode=[0.1]*len(l));
plt.title('Percentage of Embarked')
plt.legend();
df['Cabin_name'].unique()
k4=df['Cabin_name'].value_counts()
k4
sns.countplot(x='Cabin_name',data=df)
k4 = [20, 30, 10, 15, 25, 5, 18, 12]  # Replace these counts with your actual data
labels = ['B', 'C', 'E', 'G', 'D', 'A', 'F', 'T']

plt.pie(k4, labels=labels, autopct="%0.0f%%", explode=[0.1]*len(labels), colors=sns.color_palette('Set2'))
plt.title('Percentage of Categories')
plt.legend()
plt.show()
df['Name'].unique()
df['saluation']=df['Name'].str.split(",",expand=True)[1].str.split(".",expand=True)[0]
df['saluation'].unique()
df['saluation'].nunique()
df['saluation'].value_counts()
plt.figure(figsize=(25,10),dpi=200)
sns.countplot(x='saluation',data=df);
df['Family_members']=df['SibSp']+df['Parch']
df['Family_members'].unique()
sns.countplot(x='Family_members',data=df);
df['Age'].agg(['max',"min","mean"])
sns.histplot(x='Age',data=df,kde=True);
df["Age_category"]=pd.cut(df.Age,[0,14,25,60,np.inf],labels=["children","youth","adults","older"])
l1=df['Age_category'].value_counts()
l1
l1 = [20, 30, 25, 15]  # Replace these counts with your actual data
labels = ['youth', 'adults', 'children', 'older']

plt.pie(l1, labels=labels, autopct="%0.0f%%", explode=[0.1]*len(labels), hatch=['**O', 'oO', 'O.O', '.||.'])
plt.title('Age Category Distribution')
plt.legend()
plt.show()
df.groupby('Age_category')['Fare'].sum()
sns.barplot(x='Age_category',y='Fare',data=df,estimator=sum,palette='cubehelix')
df["Fare"].agg(["max","min","mean"])
sns.distplot(df["Fare"]);
age_su=df[df["Survived"]==1]["Age"]
age_nsu=df[df["Survived"]==0]["Age"]
sns.distplot(age_su,label="survived")
sns.distplot(age_nsu,label="not_survived")
plt.legend();
sns.scatterplot(x="Age",y="Fare",data=df);
df.groupby('Age_category')['Sex'].value_counts()
ax=sns.countplot(x='Age_category',data=df,hue='Sex')
ax.bar_label(ax.containers[0])
df.groupby("Survived")[["Sex","Pclass"]].value_counts()
    df['Survived'].unique()
