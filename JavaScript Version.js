// This tests the model accuracy according to the amount of data its been given
// example results: 0.5, 1.0, 1.5, 0.2223333

const music_data = require('music.csv');
const { DecisionTreeClassifier } = require('sklearn.tree');
const { train_test_split } = require('sklearn.model_selection');
const { accuracy_score } = require('sklearn.metrics');

const X = music_data.drop(columns=["genre"]);
const y = music_data["genre"];
const [X_train, X_test, y_train, y_test] = train_test_split(X, y, { test_size: 0.2 });

const model = new DecisionTreeClassifier();

model.fit(X_train, y_train);
const predictions = model.predict(X_test);

const score = accuracy_score(y_test, predictions);
score;

########################################################################################################################################################################################################


// This code recommends a music genre to a user depending on their gender. 0-female, 1-male
// [21, 1] = [21 years old, male]
// [22, 1] = [22 years old, female]

// Example Result: ['HipHop', 'Dance']
const music_data = require('music.csv');
const { DecisionTreeClassifier } = require('sklearn.tree');

const X = music_data.drop(columns=["genre"]);
const y = music_data["genre"];

const model = new DecisionTreeClassifier();

model.fit(X.values, y);
const predictions = model.predict([[21, 1], [22, 0]]);
predictions;

########################################################################################################################################################################################################
//An even shorter version of the code above
//Returns the same output

const { DecisionTreeClassifier } = require('sklearn.tree');

// Load data and create model
const model = new DecisionTreeClassifier().fit(X, df['genre']);

// Make predictions
model.predict(X.slice(0, 2));
