package rf

import (
	"math"
)

// DecisionTree represents a basic decision tree
type DecisionTree struct {
	FeatureIndex int
	Threshold    float64
	Left         *DecisionTree
	Right        *DecisionTree
	LeafValue    float64
}

func (tree *DecisionTree) Predict(features []float64) float64 {
	if tree.Left == nil && tree.Right == nil {
		return tree.LeafValue
	}
	if features[tree.FeatureIndex] <= tree.Threshold {
		return tree.Left.Predict(features)
	}
	return tree.Right.Predict(features)
}

// CalculateGini calculates the Gini impurity of a dataset
func CalculateGini(Y []float64) float64 {
	total := float64(len(Y))
	classCounts := make(map[float64]float64)
	for _, label := range Y {
		classCounts[label]++
	}

	gini := 1.0
	for _, count := range classCounts {
		prob := count / total
		gini -= prob * prob
	}
	return gini
}

func FindBestSplit(X [][]float64, Y []float64) (int, float64) {
	bestGini := math.Inf(1)
	var bestFeature int
	var bestThreshold float64

	for i := 0; i < len(X[0]); i++ {
		thresholds := make(map[float64]struct{})
		for _, features := range X {
			if i < len(features) {
				thresholds[features[i]] = struct{}{}
			}
		}

		for threshold := range thresholds {
			_, leftY, _, rightY := splitDataset(X, Y, i, threshold)
			if len(leftY) == 0 || len(rightY) == 0 {
				continue
			}
			leftGini := CalculateGini(leftY)
			rightGini := CalculateGini(rightY)
			weightedGini := (float64(len(leftY))/float64(len(Y)))*leftGini + (float64(len(rightY))/float64(len(Y)))*rightGini

			if weightedGini < bestGini {
				bestGini = weightedGini
				bestFeature = i
				bestThreshold = threshold
			}
		}
	}
	return bestFeature, bestThreshold
}

// splitDataset splits the dataset based on a feature and threshold
func splitDataset(X [][]float64, Y []float64, featureIndex int, threshold float64) ([][]float64, []float64, [][]float64, []float64) {
	var leftX, rightX [][]float64
	var leftY, rightY []float64

	if featureIndex < 0 || featureIndex >= len(X[0]) {
		// If featureIndex is out of bounds, return empty slices
		return leftX, leftY, rightX, rightY
	}

	for i, features := range X {
		if features[featureIndex] <= threshold {
			leftX = append(leftX, features)
			leftY = append(leftY, Y[i])
		} else {
			rightX = append(rightX, features)
			rightY = append(rightY, Y[i])
		}
	}
	return leftX, leftY, rightX, rightY
}

// TrainDecisionTree trains a decision tree on the data
func TrainDecisionTree(X [][]float64, Y []float64) *DecisionTree {
	if len(Y) == 0 {
		return &DecisionTree{}
	}

	feature, threshold := FindBestSplit(X, Y)
	if feature == -1 { // No valid split found
		return &DecisionTree{LeafValue: majorityVote(Y)}
	}

	leftX, leftY, rightX, rightY := splitDataset(X, Y, feature, threshold)
	if len(leftY) == 0 || len(rightY) == 0 {
		return &DecisionTree{LeafValue: majorityVote(Y)}
	}

	return &DecisionTree{
		FeatureIndex: feature,
		Threshold:    threshold,
		Left:         TrainDecisionTree(leftX, leftY),
		Right:        TrainDecisionTree(rightX, rightY),
	}
}

// majorityVote returns the majority class label from a set of labels
func majorityVote(Y []float64) float64 {
	counts := make(map[float64]int)
	for _, label := range Y {
		counts[label]++
	}

	var majority float64
	maxCount := -1
	for label, count := range counts {
		if count > maxCount {
			majority = label
			maxCount = count
		}
	}
	return majority
}
