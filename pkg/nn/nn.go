package nn

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

// Función de activación sigmoide
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivada de la función sigmoide
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Red neuronal con una capa oculta
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

// Inicializar la red neuronal
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

	// Inicialización aleatoria de los pesos
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

// Propagación hacia adelante
func (nn *NeuralNetwork) Forward(inputs []float64) []float64 {
	// Capa oculta
	for i := 0; i < nn.hiddenNeurons; i++ {
		sum := 0.0
		for j := 0; j < nn.inputs; j++ {
			sum += inputs[j] * nn.weightsInput[j][i]
		}
		nn.hiddenLayer[i] = sigmoid(sum)
	}

	// Capa de salida
	for i := 0; i < nn.outputs; i++ {
		sum := 0.0
		for j := 0; j < nn.hiddenNeurons; j++ {
			sum += nn.hiddenLayer[j] * nn.weightsOutput[j][i]
		}
		nn.outputLayer[i] = sigmoid(sum)
	}

	return nn.outputLayer
}

// Retropropagación
func (nn *NeuralNetwork) Backpropagation(inputs, expected []float64) {
	// Cálculo del error de la capa de salida
	outputErrors := make([]float64, nn.outputs)
	for i := range nn.outputLayer {
		outputErrors[i] = expected[i] - nn.outputLayer[i]
	}

	// Ajustar pesos de la capa de salida
	for i := 0; i < nn.hiddenNeurons; i++ {
		for j := 0; j < nn.outputs; j++ {
			delta := outputErrors[j] * sigmoidDerivative(nn.outputLayer[j])
			nn.weightsOutput[i][j] += nn.learningRate * delta * nn.hiddenLayer[i]
		}
	}

	// Cálculo del error de la capa oculta
	hiddenErrors := make([]float64, nn.hiddenNeurons)
	for i := 0; i < nn.hiddenNeurons; i++ {
		errorSum := 0.0
		for j := 0; j < nn.outputs; j++ {
			errorSum += outputErrors[j] * nn.weightsOutput[i][j]
		}
		hiddenErrors[i] = errorSum
	}

	// Ajustar pesos de la capa de entrada
	for i := 0; i < nn.inputs; i++ {
		for j := 0; j < nn.hiddenNeurons; j++ {
			delta := hiddenErrors[j] * sigmoidDerivative(nn.hiddenLayer[j])
			nn.weightsInput[i][j] += nn.learningRate * delta * inputs[i]
		}
	}
}

// Entrenamiento
func (nn *NeuralNetwork) Train(inputs, expected []float64) {
	nn.Forward(inputs)
	nn.Backpropagation(inputs, expected)
}
