package svm

import (
	"sync"
)

// TrainConcurrently trains the SVM model using goroutines
func (s *SVM) TrainConcurrently(X [][]float64, Y []float64) {
	numSamples := len(X)
	numFeatures := len(X[0])
	s.Weights = make([]float64, numFeatures)

	var wg sync.WaitGroup
	mutex := &sync.Mutex{}

	// Initialize slices for temporary updates
	weightUpdates := make([][]float64, numSamples)
	biasUpdates := make([]float64, numSamples)

	for i := 0; i < s.Iterations; i++ {
		// Reset temporary updates for each iteration
		for j := range weightUpdates {
			weightUpdates[j] = make([]float64, numFeatures)
		}

		for j := 0; j < numSamples; j++ {
			wg.Add(1)
			go func(index int) {
				defer wg.Done()
				dot := s.dotProduct(s.Weights, X[index])
				if Y[index]*dot <= 0 {
					// Calculate updates
					weightChange := make([]float64, numFeatures)
					for k := 0; k < numFeatures; k++ {
						weightChange[k] = s.LearningRate * Y[index] * X[index][k]
					}
					biasChange := s.LearningRate * Y[index]

					// Store updates in temporary slices
					mutex.Lock()
					weightUpdates[index] = weightChange
					biasUpdates[index] = biasChange
					mutex.Unlock()
				}
			}(j)
		}
		wg.Wait()

		// Apply accumulated updates
		mutex.Lock()
		for j := 0; j < numSamples; j++ {
			if weightUpdates[j] != nil {
				for k := 0; k < numFeatures; k++ {
					s.Weights[k] += weightUpdates[j][k]
				}
				s.Bias += biasUpdates[j]
			}
		}
		mutex.Unlock()
	}
}
