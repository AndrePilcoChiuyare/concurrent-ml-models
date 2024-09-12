package main

import (
	"PC2/pkg/nn"
	"PC2/pkg/rf"
	"PC2/pkg/svm"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func applyThreshold(output float64) int {
	if output >= 0.5 {
		return 1
	}
	return -1
}

// Convert labels
func convertLabel(label string) float64 {
	if label == "0" {
		return -1
	}
	return 1
}

// Convert strings to float64 slice
func convertFeatures(features []string) ([]float64, error) {
	var result []float64
	for _, feature := range features {
		val, err := strconv.ParseFloat(feature, 64)
		if err != nil {
			return nil, err
		}
		result = append(result, val)
	}
	return result, nil
}

// Create train and test datasets
func trainTestSplit(X [][]float64, Y []float64, testSize float64) ([][]float64, []float64, [][]float64, []float64) {
	rand.Seed(time.Now().UnixNano())
	indices := rand.Perm(len(X))

	splitIdx := int(float64(len(X)) * (1 - testSize))

	XTrain := make([][]float64, splitIdx)
	YTrain := make([]float64, splitIdx)
	XTest := make([][]float64, len(X)-splitIdx)
	YTest := make([]float64, len(X)-splitIdx)

	for i, idx := range indices {
		if i < splitIdx {
			XTrain[i] = X[idx]
			YTrain[i] = Y[idx]
		} else {
			XTest[i-splitIdx] = X[idx]
			YTest[i-splitIdx] = Y[idx]
		}
	}

	return XTrain, YTrain, XTest, YTest
}

// Calculate accuracy
func calculateAccuracy(predictions, actuals []float64) float64 {
	correct := 0
	for i := range predictions {
		if predictions[i] == actuals[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(predictions))
}

// Read CSV
func readCSV(path string) (X [][]float64, Y []float64, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	_, err = reader.Read() // Skip header
	if err != nil {
		return nil, nil, fmt.Errorf("error reading header: %v", err)
	}

	for {
		record, err := reader.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, nil, fmt.Errorf("error reading record: %v", err)
		}

		features, err := convertFeatures(record[:len(record)-1])
		if err != nil {
			return nil, nil, fmt.Errorf("error converting features: %v", err)
		}

		label := convertLabel(record[len(record)-1])
		X = append(X, features)
		Y = append(Y, label)
	}

	return X, Y, nil
}

func convertYToMatrix(Y []float64) [][]float64 {
	result := make([][]float64, len(Y))
	for i := range Y {
		result[i] = []float64{Y[i]}
	}
	return result
}

func main() {
	// Load dataset
	path := "dataset/clf_num_Higgs.csv"

	X, Y, err := readCSV(path)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Split data
	XTrain, YTrain, XTest, YTest := trainTestSplit(X, Y, 0.2)

	var models []interface{}
	var accuracies [][]float64
	// var predictions []interface{}
	var times [][]time.Duration

	// SVM -------------------------------------------------
	var svmModels []*svm.SVM
	var svmPredictions [][]float64
	var svmAccuracies []float64
	var svmTimes []time.Duration

	// SVM models
	svmSequentialModel := svm.NewSVM(0.1, 100)
	svmConcurrentModel := svm.NewSVM(0.1, 100)
	svmModels = append(svmModels, svmSequentialModel, svmConcurrentModel)

	models = append(models, svmModels)

	// Training times
	svmStartSeq := time.Now()
	svmModels[0].Train(XTrain, YTrain)
	svmElapsedSeq := time.Since(svmStartSeq)

	svmStartCon := time.Now()
	svmModels[1].TrainConcurrently(XTrain, YTrain)
	svmElapsedCon := time.Since(svmStartCon)

	svmTimes = append(svmTimes, svmElapsedSeq, svmElapsedCon)
	times = append(times, svmTimes)

	// Predictions
	svmPredictionsSequential := svmModels[0].Predict(XTest)
	svmPredictionsConcurrent := svmModels[1].Predict(XTest)
	svmPredictions = append(svmPredictions, svmPredictionsSequential, svmPredictionsConcurrent)
	// predictions = append(predictions, svmPredictions)

	// Accuracies
	svmAccuracySequential := calculateAccuracy(svmPredictions[0], YTest)
	svmAccuracyConcurrent := calculateAccuracy(svmPredictions[1], YTest)
	svmAccuracies = append(svmAccuracies, svmAccuracySequential, svmAccuracyConcurrent)
	accuracies = append(accuracies, svmAccuracies)
	// SVM -------------------------------------------------

	// Random forest ---------------------------------------
	XTrainRF := XTrain[:1000]
	YTrainRF := YTrain[:1000]
	XTestRF := XTest[:100]
	YTestRF := YTest[:100]

	var rfModels []*rf.RandomForest
	// var rfPredictions [][]float64
	var rfAccuracies []float64
	var rfTimes []time.Duration

	// Random Forest models
	rfSequentialModel := rf.NewRandomForest(10)
	rfConcurrentModel := rf.NewRandomForest(10)
	rfModels = append(rfModels, rfSequentialModel, rfConcurrentModel)

	models = append(models, rfModels)

	// Training times
	rfStartSeq := time.Now()
	rfModels[0].Train(XTrainRF, YTrainRF)
	rfElapsedSeq := time.Since(rfStartSeq)

	rfStartCon := time.Now()
	rfModels[1].TrainConcurrently(XTrainRF, YTrainRF)
	rfElapsedCon := time.Since(rfStartCon)

	rfTimes = append(rfTimes, rfElapsedSeq, rfElapsedCon)
	times = append(times, rfTimes)

	// Predictions
	var rfPredictionsSequential []float64
	var rfPredictionsConcurrent []float64

	for _, x := range XTestRF {
		rfPredictionsSequential = append(rfPredictionsSequential, rfModels[0].Predict(x))
		rfPredictionsConcurrent = append(rfPredictionsConcurrent, rfModels[1].Predict(x))
	}

	// rfPredictions = append(rfPredictions, rfPredictionsSequential, rfPredictionsConcurrent)

	// Accuracies
	rfAccuracySequential := calculateAccuracy(rfPredictionsSequential, YTestRF)
	rfAccuracyConcurrent := calculateAccuracy(rfPredictionsConcurrent, YTestRF)
	rfAccuracies = append(rfAccuracies, rfAccuracySequential, rfAccuracyConcurrent)

	accuracies = append(accuracies, rfAccuracies)
	// Random forest ---------------------------------------

	// Artificial Neural Network ---------------------------------
	// NN parameters
	var nnModels []*nn.NeuralNetwork
	var nnPredictions [][]float64
	var nnAccuracies []float64
	var nnTimes []time.Duration

	// ANN models
	inputs := 24
	hiddenNeurons := 3
	outputs := 1
	learningRate := 0.5

	nnSequentialModel := nn.NewNeuralNetwork(inputs, hiddenNeurons, outputs, learningRate)
	nnConcurrentModel := nn.NewNeuralNetwork(inputs, hiddenNeurons, outputs, learningRate)
	nnModels = append(nnModels, nnSequentialModel, nnConcurrentModel)

	models = append(models, nnModels)

	// Training times
	nnStartSeq := time.Now()
	for epoch := 0; epoch < 5; epoch++ {
		for i := range XTrain {
			nnModels[0].Train(XTrain[i], []float64{YTrain[i]})
		}
	}
	nnElapsedSeq := time.Since(nnStartSeq)

	nnStartCon := time.Now()
	for epoch := 0; epoch < 5; epoch++ {
		for i := range XTrain {
			nnModels[1].Train(XTrain[i], []float64{YTrain[i]})
		}
	}
	nnElapsedCon := time.Since(nnStartCon)

	nnTimes = append(nnTimes, nnElapsedSeq, nnElapsedCon)
	times = append(times, nnTimes)

	// Predictions
	var nnPredictionsSequential []float64
	var nnPredictionsConcurrent []float64

	for i := range XTest {
		outputSeq := nnModels[0].Forward(XTest[i])
		thresholdedOutputSeq := applyThreshold(outputSeq[0])
		nnPredictionsSequential = append(nnPredictionsSequential, float64(thresholdedOutputSeq))

		outputCon := nnModels[1].Forward(XTest[i])
		thresholdedOutputCon := applyThreshold(outputCon[0])
		nnPredictionsConcurrent = append(nnPredictionsConcurrent, float64(thresholdedOutputCon))
	}

	nnPredictions = append(nnPredictions, nnPredictionsSequential, nnPredictionsConcurrent)

	// Accuracies
	nnAccuracySequential := calculateAccuracy(nnPredictions[0], YTest)
	nnAccuracyConcurrent := calculateAccuracy(nnPredictions[1], YTest)
	nnAccuracies = append(nnAccuracies, nnAccuracySequential, nnAccuracyConcurrent)
	accuracies = append(accuracies, nnAccuracies)

	// Artificial Neural Network ---------------------------------

	for i := 0; i < len(models); i++ {
		switch modelSlice := models[i].(type) {
		case []*svm.SVM:
			fmt.Println("SVM ---------------")
			// Handle slice of SVM models
			for j := 0; j < len(modelSlice); j++ {
				var implementationType string
				if j == 0 {
					implementationType = "Sequential"
				} else {
					implementationType = "Concurrent"
				}
				fmt.Printf("%s weights: %v\n", implementationType, modelSlice[j].Weights)
				fmt.Printf("%s bias: %v\n", implementationType, modelSlice[j].Bias)
				fmt.Printf("%s training time: %s\n", implementationType, times[i][j])
				fmt.Printf("%s accuracy: %.2f%%\n", implementationType, accuracies[i][j]*100)
				fmt.Println("---------------")
			}
		case []*rf.RandomForest:
			fmt.Println("Random Forest ---------------")
			// Handle slice of DNN models
			for j := 0; j < len(modelSlice); j++ {
				var implementationType string
				if j == 0 {
					implementationType = "Sequential"
				} else {
					implementationType = "Concurrent"
				}
				// fmt.Printf("%s number of trees: %d\n", implementationType, modelSlice[j].NumTrees)
				// fmt.Printf("%s feature importances: %v\n", implementationType, modelSlice[j].FeatureImportances)
				fmt.Printf("%s training time: %s\n", implementationType, times[i][j])
				fmt.Printf("%s accuracy: %.2f%%\n", implementationType, accuracies[i][j]*100)
				fmt.Println("---------------")
			}
		case []*nn.NeuralNetwork:
			fmt.Println("Artificial Neural Network ---------------")
			for j := 0; j < len(modelSlice); j++ {
				var implementationType string
				if j == 0 {
					implementationType = "Sequential"
				} else {
					implementationType = "Concurrent"
				}
				fmt.Printf("%s training time: %s\n", implementationType, times[i][j])
				fmt.Printf("%s accuracy: %.2f%%\n", implementationType, accuracies[i][j]*100)
				fmt.Println("---------------")
			}
		default:
			fmt.Println("Unknown model type")
		}
	}
}
