package rf

import (
	"sync"
)

// RandomForest represents a collection of decision trees
type RandomForest struct {
	Trees []*DecisionTree
}

// NewRandomForest initializes a new Random Forest with the given number of trees
func NewRandomForest(numTrees int) *RandomForest {
	return &RandomForest{Trees: make([]*DecisionTree, numTrees)}
}

// Train trains the Random Forest sequentially
func (rf *RandomForest) Train(X [][]float64, Y []float64) {
	for i := range rf.Trees {
		rf.Trees[i] = TrainDecisionTree(X, Y)
	}
}

// TrainConcurrently trains the Random Forest concurrently
func (rf *RandomForest) TrainConcurrently(X [][]float64, Y []float64) {
	var wg sync.WaitGroup
	mutex := &sync.Mutex{}

	for i := range rf.Trees {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			tree := TrainDecisionTree(X, Y)
			mutex.Lock()
			rf.Trees[index] = tree
			mutex.Unlock()
		}(i)
	}
	wg.Wait()
}

// Predict makes predictions using the Random Forest
func (rf *RandomForest) Predict(features []float64) float64 {
	votes := make(map[float64]int)
	for _, tree := range rf.Trees {
		vote := tree.Predict(features)
		votes[vote]++
	}

	var maxVote float64
	maxCount := -1
	for vote, count := range votes {
		if count > maxCount {
			maxCount = count
			maxVote = vote
		}
	}
	return maxVote
}
