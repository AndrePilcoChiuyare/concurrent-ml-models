package dnn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Activation function (Sigmoid)
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Sigmoid derivative for backpropagation
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Neural Network struct with multiple layers
type NeuralNetwork struct {
	layerSizes   []int         // List of sizes of layers
	weights      [][][]float64 // Weights between each layer
	biases       [][]float64   // Biases for each layer
	learningRate float64
}

// Initialize the neural network with random weights and biases
func (nn *NeuralNetwork) Initialize(layerSizes []int, learningRate float64) {
	nn.layerSizes = layerSizes
	nn.learningRate = learningRate

	// Initialize weights and biases for each layer
	nn.weights = make([][][]float64, len(layerSizes)-1)
	nn.biases = make([][]float64, len(layerSizes)-1)

	rand.Seed(time.Now().UnixNano())
	for l := 0; l < len(layerSizes)-1; l++ {
		nn.weights[l] = make([][]float64, layerSizes[l])
		for i := range nn.weights[l] {
			nn.weights[l][i] = make([]float64, layerSizes[l+1])
			for j := range nn.weights[l][i] {
				nn.weights[l][i][j] = rand.Float64()*2 - 1 // Random values between -1 and 1
			}
		}
		nn.biases[l] = make([]float64, layerSizes[l+1])
		for i := range nn.biases[l] {
			nn.biases[l][i] = rand.Float64()*2 - 1
		}
	}
}

// Forward propagation
func (nn *NeuralNetwork) Forward(input []float64) ([][]float64, []float64) {
	activations := make([][]float64, len(nn.layerSizes))
	activations[0] = input

	// Calculate activations for each layer
	for l := 0; l < len(nn.layerSizes)-1; l++ {
		nextLayer := make([]float64, nn.layerSizes[l+1])
		for j := 0; j < nn.layerSizes[l+1]; j++ {
			activation := nn.biases[l][j]
			for i := 0; i < nn.layerSizes[l]; i++ {
				activation += activations[l][i] * nn.weights[l][i][j]
			}
			nextLayer[j] = sigmoid(activation)
		}
		activations[l+1] = nextLayer
	}
	return activations, activations[len(activations)-1]
}

// Backpropagation and weight update
func (nn *NeuralNetwork) Backpropagation(input []float64, target float64) {
	activations, output := nn.Forward(input)

	// Output layer error
	deltas := make([][]float64, len(nn.layerSizes)-1)
	deltas[len(deltas)-1] = make([]float64, nn.layerSizes[len(nn.layerSizes)-1])
	for j := 0; j < nn.layerSizes[len(nn.layerSizes)-1]; j++ {
		error := target - output[j]
		deltas[len(deltas)-1][j] = error * sigmoidDerivative(output[j])
	}

	// Backpropagate the error
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

	// Update weights and biases
	for l := 0; l < len(nn.layerSizes)-1; l++ {
		for i := 0; i < nn.layerSizes[l]; i++ {
			for j := 0; j < nn.layerSizes[l+1]; j++ {
				nn.weights[l][i][j] += nn.learningRate * deltas[l][j] * activations[l][i]
			}
		}
		for j := 0; j < nn.layerSizes[l+1]; j++ {
			nn.biases[l][j] += nn.learningRate * deltas[l][j]
		}
	}
}

// Train the network
func (nn *NeuralNetwork) Train(inputs [][]float64, targets []float64, epochs int) {
	fmt.Println("DNN Sequential training")
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		for i := 0; i < len(inputs); i++ {
			_, output := nn.Forward(inputs[i])
			nn.Backpropagation(inputs[i], targets[i])
			totalError += math.Pow(targets[i]-output[0], 2)
		}
		fmt.Printf("Epoch: %d, Error: %.5f\n", epoch, totalError/float64(len(inputs)))
	}
}
