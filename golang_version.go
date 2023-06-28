package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/linear_models"
)

func main() {
	// Load data
	file, err := os.Open("music.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Prepare data
	var instances [][]string
	for _, record := range records {
		instances = append(instances, record[1:3])
	}

	// Convert data to GoLearn format
	instancesData := base.ParseCSVToInstances(instances, true)
	instancesData.SetClassIndex(1)

	// Split data into features and target
	X, y := base.InstancesTrainTestSplit(instancesData, 1.0)

	// Create model and fit to data
	model := linear_models.NewLogisticRegression()
	model.Fit(X, y)

	// Make predictions
	preds, err := model.Predict(X)
	if err != nil {
		log.Fatal(err)
	}

	// Print predictions
	for _, pred := range preds {
		fmt.Println(pred)
	}
}
