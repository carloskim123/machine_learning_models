

// Import required libraries
const pandas = require('pandas');
const DecisionTreeClassifier = require('sklearn.tree.DecisionTreeClassifier');

// Load data
const data_frame = pandas.read_csv('music.csv');

// Split data into features and target
const X = data_frame.drop(columns=['genre']);
const y = data_frame['genre'];

// Create model and fit to data
const model = new DecisionTreeClassifier();
model.fit(X, y);

// Make predictions
const preds = model.predict([[21, 1], [22, 0]]);
preds;
