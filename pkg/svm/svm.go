package svm

// SVM represents a simple linear SVM model
type SVM struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Iterations   int
}

// NewSVM creates a new SVM model with given parameters
func NewSVM(learningRate float64, iterations int) *SVM {
	return &SVM{
		LearningRate: learningRate,
		Iterations:   iterations,
	}
}

// Train trains the SVM model using gradient descent
func (s *SVM) Train(X [][]float64, Y []float64) {
	numSamples := len(X)
	numFeatures := len(X[0])
	s.Weights = make([]float64, numFeatures)

	for i := 0; i < s.Iterations; i++ {
		for j := 0; j < numSamples; j++ {
			dot := s.dotProduct(s.Weights, X[j])
			if Y[j]*dot <= 0 {
				// Update weights and bias
				for k := 0; k < numFeatures; k++ {
					s.Weights[k] += s.LearningRate * Y[j] * X[j][k]
				}
				s.Bias += s.LearningRate * Y[j]
			}
		}
	}
}

// Predict predicts the class for given input data
func (s *SVM) Predict(X [][]float64) []float64 {
	numSamples := len(X)
	predictions := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		dot := s.dotProduct(s.Weights, X[i]) + s.Bias
		if dot >= 0 {
			predictions[i] = 1
		} else {
			predictions[i] = -1
		}
	}

	return predictions
}

// dotProduct calculates the dot product of two vectors
func (s *SVM) dotProduct(vec1, vec2 []float64) float64 {
	sum := 0.0
	for i := 0; i < len(vec1); i++ {
		sum += vec1[i] * vec2[i]
	}
	return sum
}
