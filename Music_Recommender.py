# This tests the model accuracy according to the amount of data its been given
# example results: 0.5, 1.0, 1.5, 0.2223333

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv("music.csv")
X = music_data.drop(columns=["genre"])
y = music_data["genre"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
score

########################################################################################################################################################################################################


# This code recommends a music genre to a user depending on their gender. 0-female, 1-male
# [21, 1] = [21 years old, male]
# [22, 1] = [22 years old, female]

# Example Result: array(['HipHop', 'Dance'], dtype=object)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv("music.csv")
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

model = DecisionTreeClassifier()

model.fit(X.values, y)
predictions = model.predict([[21, 1], [22, 0]])
predictions


########################################################################################################################################################################################################
#An even shorter version of the code above
#Returns the same output

from sklearn.tree import DecisionTreeClassifier

# Load data and create model
model = DecisionTreeClassifier().fit(X, df['genre'])

# Make predictions
model.predict(X[:2])


