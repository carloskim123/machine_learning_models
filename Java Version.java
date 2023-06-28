// This tests the model accuracy according to the amount of data its been given
// example results: 0.5, 1.0, 1.5, 0.2223333

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MusicGenreClassifier {
    public static void main(String[] args) throws Exception {
        // Load data
        Instances musicData = loadData("music.csv");

        // Split data into training and testing sets
        Instances trainData = musicData.trainCV(5, 0);
        Instances testData = musicData.testCV(5, 0);

        // Create and build the classifier
        Classifier classifier = new weka.classifiers.trees.J48();
        classifier.buildClassifier(trainData);

        // Evaluate the classifier
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifier, testData);

        // Print the accuracy score
        System.out.println(eval.pctCorrect());
    }

    private static Instances loadData(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        List<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("age"));
        attributes.add(new Attribute("gender"));
        attributes.add(new Attribute("genre", (List<String>) null));

        Instances data = new Instances("music_data", attributes, 0);

        String line;
        while ((line = reader.readLine()) != null) {
            String[] values = line.split(",");
            Instance instance = new DenseInstance(3);
            instance.setValue(attributes.get(0), Double.parseDouble(values[0]));
            instance.setValue(attributes.get(1), Double.parseDouble(values[1]));
            instance.setValue(attributes.get(2), values[2]);
            data.add(instance);
        }

        reader.close();

        return data;
    }
}
