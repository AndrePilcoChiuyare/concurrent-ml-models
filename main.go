package main

import (
	"PC2/pkg/ann"
	"PC2/pkg/cf"
	"PC2/pkg/dnn"
	"PC2/pkg/lfm"
	"PC2/pkg/rf"
	"PC2/pkg/svm"
	"encoding/csv"
	"fmt"
	"io"
	"log"
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
	var nnModels []*ann.NeuralNetwork
	var nnPredictions [][]float64
	var nnAccuracies []float64
	var nnTimes []time.Duration

	// ANN models
	inputs := 24
	hiddenNeurons := 3
	outputs := 1
	learningRate := 0.5

	nnSequentialModel := ann.NewNeuralNetwork(inputs, hiddenNeurons, outputs, learningRate)
	nnConcurrentModel := ann.NewNeuralNetwork(inputs, hiddenNeurons, outputs, learningRate)
	nnModels = append(nnModels, nnSequentialModel, nnConcurrentModel)

	models = append(models, nnModels)

	// Training times
	nnStartSeq := time.Now()
	for epoch := 0; epoch < 100; epoch++ {
		for i := range XTrain {
			nnModels[0].Train(XTrain[i], []float64{YTrain[i]})
		}
	}
	nnElapsedSeq := time.Since(nnStartSeq)

	nnStartCon := time.Now()
	for epoch := 0; epoch < 100; epoch++ {
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

	// Deep Neural Network ---------------------------------------
	// Initialize neural network (input layer with 2 neurons, 2 hidden layers with 4 and 3 neurons, and output layer with 1 neuron)
	var dnnModels []*dnn.NeuralNetwork
	var dnnAccuracies []float64
	var dnnTimes []time.Duration

	// DNN models
	dnnSequentialModel := dnn.NeuralNetwork{}
	dnnSequentialModel.Initialize([]int{24, 4, 3, 1}, 0.01)
	dnnConcurrentModel := dnn.NeuralNetwork{}
	dnnConcurrentModel.Initialize([]int{24, 4, 3, 1}, 0.01)
	dnnModels = append(dnnModels, &dnnSequentialModel, &dnnConcurrentModel)

	models = append(models, dnnModels)

	// Training times
	dnnStartSeq := time.Now()
	dnnSequentialModel.Train(XTrain, YTrain, 3)
	dnnElapsedSeq := time.Since(dnnStartSeq)

	dnnStartCon := time.Now()
	dnnConcurrentModel.TrainConcurrently(XTrain, YTrain, 3, 2048)
	dnnElapsedCon := time.Since(dnnStartCon)

	dnnTimes = append(dnnTimes, dnnElapsedSeq, dnnElapsedCon)
	times = append(times, dnnTimes)

	// Predictions
	var dnnPredictionsSequential []float64
	var dnnPredictionsConcurrent []float64

	for i := range XTest {
		_, outputSeq := dnnSequentialModel.Forward(XTest[i])
		thresholdedOutputSeq := applyThreshold(outputSeq[0])
		dnnPredictionsSequential = append(dnnPredictionsSequential, float64(thresholdedOutputSeq))

		_, outputCon := dnnConcurrentModel.Forward(XTest[i])
		thresholdedOutputCon := applyThreshold(outputCon[0])
		dnnPredictionsConcurrent = append(dnnPredictionsConcurrent, float64(thresholdedOutputCon))
	}

	// Accuracies
	dnnAccuracySequential := calculateAccuracy(dnnPredictionsSequential, YTest)
	dnnAccuracyConcurrent := calculateAccuracy(dnnPredictionsConcurrent, YTest)
	dnnAccuracies = append(dnnAccuracies, dnnAccuracySequential, dnnAccuracyConcurrent)
	accuracies = append(accuracies, dnnAccuracies)
	// Deep Neural Network ---------------------------------------

	// Collaborative filtering --------------------------------------
	// Load data from CSV
	userRatings, err := cf.LoadCSV("dataset/rating.csv")
	if err != nil {
		log.Fatalf("Error loading CSV: %v", err)
	}

	// Example userId for whom we want recommendations
	targetUserId := 1
	cfStartSeq := time.Now()
	recommendationsSeq := cf.RecommendMovies(targetUserId, userRatings)
	cfElapsedSeq := time.Since(cfStartSeq)

	cfStartCon := time.Now()
	recommendationsCon := cf.RecommendMoviesConcurrent(targetUserId, userRatings)
	cfElapsedCon := time.Since(cfStartCon)

	fmt.Println("Colaborative filtering ---------------")
	fmt.Printf("Recommended Movies (Sequential) for %d: %d\n", targetUserId, recommendationsSeq[:3])
	fmt.Printf("Sequential recommendation time: %s\n", cfElapsedSeq)
	fmt.Printf("Recommended Movies (Concurrent) for %d: %d\n", targetUserId, recommendationsCon[:3])
	fmt.Printf("Concurrent recommendation time: %s\n", cfElapsedCon)
	fmt.Println("---------------")
	// Collaborative filtering --------------------------------------

	// Latent factor model (Matrix factorization) -------------------
	lfmRatings, numUsers, numItems, err := lfm.LoadCSV("dataset/rating.csv")
	if err != nil {
		log.Fatalf("Error loading CSV: %v", err)
	}

	lfmModelSeq := lfm.NewLatentFactorModel(numUsers+1, numItems+1, 10, 0.01, 0.1)
	lfmModelCon := lfm.NewConcurrentLatentFactorModel(numUsers+1, numItems+1, 10, 0.01, 0.1)

	lfmStartSeq := time.Now()
	lfmModelSeq.Train(lfmRatings, 20)
	lfmElapsedSeq := time.Since(lfmStartSeq)

	lfmStartCon := time.Now()
	lfmModelCon.Train(lfmRatings, 20)
	lfmElapsedCon := time.Since(lfmStartCon)

	// Example prediction
	userId := 1
	itemId := 2
	predictedRatingSeq := lfmModelSeq.Predict(userId, itemId)
	predictedRatingCon := lfmModelCon.Predict(userId, itemId)

	fmt.Println("Matrix factorization ---------------")
	fmt.Printf("Predicted rating (Sequential) for user %d and item %d: %f\n", userId, itemId, predictedRatingSeq)
	fmt.Printf("Sequential training time: %s\n", lfmElapsedSeq)
	fmt.Printf("Predicted rating (Concurrent) for user %d and item %d: %f\n", userId, itemId, predictedRatingCon)
	fmt.Printf("Concurrent training time: %s\n", lfmElapsedCon)
	fmt.Println("---------------")
	// Latent factor model (Matrix factorization) -------------------

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
		case []*ann.NeuralNetwork:
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
		case []*dnn.NeuralNetwork:
			fmt.Println("Deep Neural Network ---------------")
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
