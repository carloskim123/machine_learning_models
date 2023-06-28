import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MusicClassifier {
    public static void main(String[] args) {
        // Load data
        Instances data = loadData("music.csv");

        // Split data into features and target
        data.setClassIndex(data.numAttributes() - 1);
        Instances features = new Instances(data);
        features.deleteAttributeAt(data.numAttributes() - 1);

        // Create model and fit to data
        J48 model = new J48();
        try {
            model.buildClassifier(data);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Make predictions
        Instance instance1 = createInstance(features, 21, 1);
        Instance instance2 = createInstance(features, 22, 0);
        double[] preds;
        try {
            preds = model.distributionForInstance(instance1);
            System.out.println("Prediction for instance 1: " + data.classAttribute().value((int) preds[0]));
            preds = model.distributionForInstance(instance2);
            System.out.println("Prediction for instance 2: " + data.classAttribute().value((int) preds[0]));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Instances loadData(String filename) {
        BufferedReader reader;
        Instances data = null;
        try {
            reader = new BufferedReader(new FileReader(filename));
            data = new Instances(reader);
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    private static Instance createInstance(Instances data, int age, int gender) {
        List<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("age"));
        attributes.add(new Attribute("gender"));

        Instances instances = new Instances("instance", attributes, 1);
        instances.setClassIndex(data.numAttributes() - 1);

        Instance instance = new DenseInstance(2);
        instance.setDataset(instances);
        instance.setValue(0, age);
        instance.setValue(1, gender);

        return instance;
    }
}
