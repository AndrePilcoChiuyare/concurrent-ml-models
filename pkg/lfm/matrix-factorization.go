package lfm

import (
	"encoding/csv"
	"math/rand"
	"os"
	"strconv"
)

// LatentFactorModel represents a basic latent factor model
type LatentFactorModel struct {
	UserFactors    matrix
	ItemFactors    matrix
	LearningRate   float64
	Regularization float64
	NumFactors     int
	NumUsers       int
	NumItems       int
}

// matrix is a 2D slice of float64
type matrix [][]float64

// NewLatentFactorModel initializes a new LatentFactorModel
func NewLatentFactorModel(numUsers, numItems, numFactors int, learningRate, regularization float64) *LatentFactorModel {
	return &LatentFactorModel{
		UserFactors:    randomMatrix(numUsers, numFactors),
		ItemFactors:    randomMatrix(numItems, numFactors),
		LearningRate:   learningRate,
		Regularization: regularization,
		NumFactors:     numFactors,
		NumUsers:       numUsers,
		NumItems:       numItems,
	}
}

// randomMatrix creates a matrix with random values
func randomMatrix(rows, cols int) matrix {
	m := make(matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)
		for j := range m[i] {
			m[i][j] = rand.Float64() * 0.1
		}
	}
	return m
}

// Train trains the model using stochastic gradient descent
func (lfm *LatentFactorModel) Train(ratings map[int]map[int]float64, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		for user, items := range ratings {
			for item, rating := range items {
				prediction := lfm.Predict(user, item)
				error := rating - prediction

				for k := 0; k < lfm.NumFactors; k++ {
					userFactor := lfm.UserFactors[user][k]
					itemFactor := lfm.ItemFactors[item][k]

					lfm.UserFactors[user][k] += lfm.LearningRate * (error*itemFactor - lfm.Regularization*userFactor)
					lfm.ItemFactors[item][k] += lfm.LearningRate * (error*userFactor - lfm.Regularization*itemFactor)
				}
			}
		}
	}
}

// predict calculates the predicted rating for a given user and item
func (lfm *LatentFactorModel) Predict(user, item int) float64 {
	var prediction float64
	for k := 0; k < lfm.NumFactors; k++ {
		prediction += lfm.UserFactors[user][k] * lfm.ItemFactors[item][k]
	}
	return prediction
}

func LoadCSV(filename string) (map[int]map[int]float64, int, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, 0, 0, err
	}

	ratings := make(map[int]map[int]float64)
	numUsers := 0
	numItems := 0

	for _, record := range records[1:] { // Skip header
		userId, _ := strconv.Atoi(record[0])
		itemId, _ := strconv.Atoi(record[1])
		rating, _ := strconv.ParseFloat(record[2], 64)

		if _, exists := ratings[userId]; !exists {
			ratings[userId] = make(map[int]float64)
		}
		ratings[userId][itemId] = rating

		if userId > numUsers {
			numUsers = userId
		}
		if itemId > numItems {
			numItems = itemId
		}
	}
	return ratings, numUsers, numItems, nil
}
