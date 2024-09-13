package lfm

import (
	"sync"
)

// ConcurrentLatentFactorModel represents an improved concurrent latent factor model
type ConcurrentLatentFactorModel struct {
	UserFactors    matrix
	ItemFactors    matrix
	LearningRate   float64
	Regularization float64
	NumFactors     int
	NumUsers       int
	NumItems       int
	Mutex          sync.Mutex
	Updates        chan updateRequest
}

// updateRequest represents a request to update factors
type updateRequest struct {
	user   int
	item   int
	rating float64
}

// NewConcurrentLatentFactorModel initializes a new ConcurrentLatentFactorModel
func NewConcurrentLatentFactorModel(numUsers, numItems, numFactors int, learningRate, regularization float64) *ConcurrentLatentFactorModel {
	return &ConcurrentLatentFactorModel{
		UserFactors:    randomMatrix(numUsers, numFactors),
		ItemFactors:    randomMatrix(numItems, numFactors),
		LearningRate:   learningRate,
		Regularization: regularization,
		NumFactors:     numFactors,
		NumUsers:       numUsers,
		NumItems:       numItems,
		Updates:        make(chan updateRequest, 1000), // Buffered channel
	}
}

// Train trains the model using concurrent stochastic gradient descent
func (lfm *ConcurrentLatentFactorModel) Train(ratings map[int]map[int]float64, epochs int) {
	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < 4; i++ { // Adjust the number of workers based on your system's capabilities
		wg.Add(1)
		go lfm.worker(&wg)
	}

	for epoch := 0; epoch < epochs; epoch++ {
		for user, items := range ratings {
			for item, rating := range items {
				lfm.Updates <- updateRequest{user, item, rating}
			}
		}
	}

	close(lfm.Updates)
	wg.Wait()
}

// worker processes update requests from the Updates channel
func (lfm *ConcurrentLatentFactorModel) worker(wg *sync.WaitGroup) {
	defer wg.Done()
	for req := range lfm.Updates {
		lfm.updateFactors(req.user, req.item, req.rating)
	}
}

// updateFactors updates user and item factors based on a single rating
func (lfm *ConcurrentLatentFactorModel) updateFactors(user, item int, rating float64) {
	prediction := lfm.Predict(user, item)
	error := rating - prediction

	for k := 0; k < lfm.NumFactors; k++ {
		userFactor := lfm.UserFactors[user][k]
		itemFactor := lfm.ItemFactors[item][k]

		lfm.UserFactors[user][k] += lfm.LearningRate * (error*itemFactor - lfm.Regularization*userFactor)
		lfm.ItemFactors[item][k] += lfm.LearningRate * (error*userFactor - lfm.Regularization*itemFactor)
	}
}

// predict calculates the predicted rating for a given user and item
func (lfm *ConcurrentLatentFactorModel) Predict(user, item int) float64 {
	var prediction float64
	for k := 0; k < lfm.NumFactors; k++ {
		prediction += lfm.UserFactors[user][k] * lfm.ItemFactors[item][k]
	}
	return prediction
}
