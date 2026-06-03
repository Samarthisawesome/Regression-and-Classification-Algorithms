import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, r2_score,confusion_matrix
from sklearn.model_selection import train_test_split

# PREPROCESSING

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Dropping columns which are not needed
df.drop(columns=["Name", "Ticket", "Cabin", "Embarked"], inplace=True)

# Imputing the missing values in age column with median using pandas
df["Age"] = df["Age"].fillna(df["Age"].median())

# Mapping male and female values to 0 and 1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Feature selection
features = ["Pclass", "Sex", "Age", "Parch", "Fare"]
target = "Survived" 

X = df[features]
Y = df[target]

# LINEAR REGRESSION 
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X_train, Y_train)
prediction1 = model1.predict(X_test)

print("Linear Regression R2 Score:", r2_score(Y_test, prediction1))

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression()
model2.fit(X_train, Y_train)
prediction2 = model2.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(Y_test, prediction2))

# DECISION TREE
from sklearn.tree import DecisionTreeClassifier

model3 = DecisionTreeClassifier()
model3.fit(X_train, Y_train)
prediction3 = model3.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(Y_test, prediction3))

#SVM
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scale=StandardScaler()
X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.fit_transform(X_test)

model4=SVC()
model4.fit(X_train_scaled,Y_train)
prediction4=model4.predict(X_test_scaled)

print("SVM Accuracy:",accuracy_score(Y_test,prediction4))

#VISUALISATION
plt.close("all")
plt.clf()
sns.set_theme(style="whitegrid")
sns.color_palette("mako", as_cmap=True)
sns.heatmap(df.corr(),annot=True,cmap="mako",fmt=".2f")
plt.title("Correlation Heatmap of Titanic dataset features")
plt.tight_layout()
plt.show()

plt.close("all")
plt.clf()
cm=confusion_matrix(Y_test,prediction2)
sns.heatmap(cm,annot=True,fmt="d",cmap="Greens",xticklabels=['Not Survived','Survived'],yticklabels=['Not Survived','Survived'])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion matrix-Logistic Regression")
plt.tight_layout()
plt.show()
