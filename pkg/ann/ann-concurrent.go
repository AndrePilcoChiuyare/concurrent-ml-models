package ann

import (
	"sync"
)

// Concurrent backpropagation
func (nn *NeuralNetwork) BackpropagationConcurrent(inputs, expected []float64) {
	// Computing the output layer error
	outputErrors := make([]float64, nn.outputs)
	for i := range nn.outputLayer {
		outputErrors[i] = expected[i] - nn.outputLayer[i]
	}

	var wg sync.WaitGroup

	// Adjust concurrently the output layer weights
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < nn.hiddenNeurons; i++ {
			for j := 0; j < nn.outputs; j++ {
				nn.mu.Lock()
				delta := outputErrors[j] * sigmoidDerivative(nn.outputLayer[j])
				nn.weightsOutput[i][j] += nn.learningRate * delta * nn.hiddenLayer[i]
				nn.mu.Unlock()
			}
		}
	}()

	// Computing hidden layer error
	hiddenErrors := make([]float64, nn.hiddenNeurons)
	for i := 0; i < nn.hiddenNeurons; i++ {
		errorSum := 0.0
		for j := 0; j < nn.outputs; j++ {
			errorSum += outputErrors[j] * nn.weightsOutput[i][j]
		}
		hiddenErrors[i] = errorSum
	}

	// Adjusting concurrently the input layer weights
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < nn.inputs; i++ {
			for j := 0; j < nn.hiddenNeurons; j++ {
				nn.mu.Lock()
				delta := hiddenErrors[j] * sigmoidDerivative(nn.hiddenLayer[j])
				nn.weightsInput[i][j] += nn.learningRate * delta * inputs[i]
				nn.mu.Unlock()
			}
		}
	}()

	// Wait for all goroutines to finish
	wg.Wait()
}
