package dnn

import (
	"fmt"
	"math"
	"sync"
)

// Forward propagation (concurrent at the layer level)
func (nn *NeuralNetwork) ForwardConcurrent(input []float64) ([][]float64, []float64) {
	activations := make([][]float64, len(nn.layerSizes))
	activations[0] = input

	// Calculate activations for each layer concurrently
	for l := 0; l < len(nn.layerSizes)-1; l++ {
		nextLayer := make([]float64, nn.layerSizes[l+1])
		var wg sync.WaitGroup
		// Parallelize by layer rather than neuron
		wg.Add(1)
		go func(l int) {
			defer wg.Done()
			for j := 0; j < nn.layerSizes[l+1]; j++ {
				activation := nn.biases[l][j]
				for i := 0; i < nn.layerSizes[l]; i++ {
					activation += activations[l][i] * nn.weights[l][i][j]
				}
				nextLayer[j] = sigmoid(activation)
			}
		}(l)
		wg.Wait()
		activations[l+1] = nextLayer
	}
	return activations, activations[len(activations)-1]
}

// Backpropagation and weight update (parallelized at the layer level, no frequent locking)
func (nn *NeuralNetwork) BackpropagationConcurrentOptimized(input []float64, target float64) {
	activations, output := nn.ForwardConcurrent(input)

	// Output layer error
	deltas := make([][]float64, len(nn.layerSizes)-1)
	deltas[len(deltas)-1] = make([]float64, nn.layerSizes[len(nn.layerSizes)-1])
	for j := 0; j < nn.layerSizes[len(nn.layerSizes)-1]; j++ {
		error := target - output[j]
		deltas[len(deltas)-1][j] = error * sigmoidDerivative(output[j])
	}

	// Backpropagate the error (parallelized at the layer level)
	for l := len(deltas) - 2; l >= 0; l-- {
		deltas[l] = make([]float64, nn.layerSizes[l+1])
		for i := 0; i < nn.layerSizes[l+1]; i++ {
			error := 0.0
			for j := 0; j < nn.layerSizes[l+2]; j++ {
				error += deltas[l+1][j] * nn.weights[l+1][i][j]
			}
			deltas[l][i] = error * sigmoidDerivative(activations[l+1][i])
		}
	}

	// Update weights and biases (do the actual updates after calculating in parallel)
	var wg sync.WaitGroup
	for l := 0; l < len(nn.layerSizes)-1; l++ {
		wg.Add(1)
		go func(l int) {
			defer wg.Done()
			weightUpdates := make([][]float64, nn.layerSizes[l])
			for i := 0; i < nn.layerSizes[l]; i++ {
				weightUpdates[i] = make([]float64, nn.layerSizes[l+1])
				for j := 0; j < nn.layerSizes[l+1]; j++ {
					weightUpdates[i][j] = nn.learningRate * deltas[l][j] * activations[l][i]
				}
			}

			// Apply weight updates after calculation
			for i := 0; i < nn.layerSizes[l]; i++ {
				for j := 0; j < nn.layerSizes[l+1]; j++ {
					nn.weights[l][i][j] += weightUpdates[i][j]
				}
			}
			// Update biases
			for j := 0; j < nn.layerSizes[l+1]; j++ {
				nn.biases[l][j] += nn.learningRate * deltas[l][j]
			}
		}(l)
	}
	wg.Wait() // Wait for all layer updates to complete
}

// Train the network concurrently with mini-batch
func (nn *NeuralNetwork) TrainConcurrently(inputs [][]float64, targets []float64, epochs int, batchSize int) {
	fmt.Println("Concurrent training")
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		fmt.Printf("Epoch %d starting...\n", epoch)

		var wg sync.WaitGroup
		for start := 0; start < len(inputs); start += batchSize {
			end := start + batchSize
			if end > len(inputs) {
				end = len(inputs)
			}
			batchInputs := inputs[start:end]
			batchTargets := targets[start:end]

			wg.Add(1)
			go func(batchInputs [][]float64, batchTargets []float64) {
				defer wg.Done()
				for i := 0; i < len(batchInputs); i++ {
					_, output := nn.ForwardConcurrent(batchInputs[i])
					nn.Backpropagation(batchInputs[i], batchTargets[i])
					totalError += math.Pow(batchTargets[i]-output[0], 2)
				}
			}(batchInputs, batchTargets)
		}
		wg.Wait()
		fmt.Printf("Epoch: %d, Error: %.5f\n", epoch, totalError/float64(len(inputs)))
	}
}
