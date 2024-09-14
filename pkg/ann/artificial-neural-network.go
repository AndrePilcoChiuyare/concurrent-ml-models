package ann

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Sigmoid function derivative
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Neural network with 1 hidden layer
type NeuralNetwork struct {
	inputs        int
	hiddenNeurons int
	outputs       int
	weightsInput  [][]float64
	weightsOutput [][]float64
	hiddenLayer   []float64
	outputLayer   []float64
	learningRate  float64
	mu            sync.Mutex
}

// Initializing the neural network
func NewNeuralNetwork(inputs, hiddenNeurons, outputs int, learningRate float64) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())
	nn := &NeuralNetwork{
		inputs:        inputs,
		hiddenNeurons: hiddenNeurons,
		outputs:       outputs,
		learningRate:  learningRate,
		hiddenLayer:   make([]float64, hiddenNeurons),
		outputLayer:   make([]float64, outputs),
	}

	// Initializing the weights randomly
	nn.weightsInput = make([][]float64, inputs)
	for i := range nn.weightsInput {
		nn.weightsInput[i] = make([]float64, hiddenNeurons)
		for j := range nn.weightsInput[i] {
			nn.weightsInput[i][j] = rand.Float64()
		}
	}

	nn.weightsOutput = make([][]float64, hiddenNeurons)
	for i := range nn.weightsOutput {
		nn.weightsOutput[i] = make([]float64, outputs)
		for j := range nn.weightsOutput[i] {
			nn.weightsOutput[i][j] = rand.Float64()
		}
	}

	return nn
}

// Forward propagation
func (nn *NeuralNetwork) Forward(inputs []float64) []float64 {
	// Capa oculta
	for i := 0; i < nn.hiddenNeurons; i++ {
		sum := 0.0
		for j := 0; j < nn.inputs; j++ {
			sum += inputs[j] * nn.weightsInput[j][i]
		}
		nn.hiddenLayer[i] = sigmoid(sum)
	}

	// Output layer
	for i := 0; i < nn.outputs; i++ {
		sum := 0.0
		for j := 0; j < nn.hiddenNeurons; j++ {
			sum += nn.hiddenLayer[j] * nn.weightsOutput[j][i]
		}
		nn.outputLayer[i] = sigmoid(sum)
	}

	return nn.outputLayer
}

// Backpropagation
func (nn *NeuralNetwork) Backpropagation(inputs, expected []float64) {
	// Computing the output layer error
	outputErrors := make([]float64, nn.outputs)
	for i := range nn.outputLayer {
		outputErrors[i] = expected[i] - nn.outputLayer[i]
	}

	// Adjusting output layer error
	for i := 0; i < nn.hiddenNeurons; i++ {
		for j := 0; j < nn.outputs; j++ {
			delta := outputErrors[j] * sigmoidDerivative(nn.outputLayer[j])
			nn.weightsOutput[i][j] += nn.learningRate * delta * nn.hiddenLayer[i]
		}
	}

	// Compute hidden layer error
	hiddenErrors := make([]float64, nn.hiddenNeurons)
	for i := 0; i < nn.hiddenNeurons; i++ {
		errorSum := 0.0
		for j := 0; j < nn.outputs; j++ {
			errorSum += outputErrors[j] * nn.weightsOutput[i][j]
		}
		hiddenErrors[i] = errorSum
	}

	// Adjust input layer error
	for i := 0; i < nn.inputs; i++ {
		for j := 0; j < nn.hiddenNeurons; j++ {
			delta := hiddenErrors[j] * sigmoidDerivative(nn.hiddenLayer[j])
			nn.weightsInput[i][j] += nn.learningRate * delta * inputs[i]
		}
	}
}

// Training
func (nn *NeuralNetwork) Train(inputs, expected []float64) {
	nn.Forward(inputs)
	nn.Backpropagation(inputs, expected)
}
