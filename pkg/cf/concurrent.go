package cf

import (
	"sync"
)

// Recommend movies based on user ratings using concurrency
func RecommendMoviesConcurrent(userId int, userRatings UserRatings) []int {
	var wg sync.WaitGroup
	mu := &sync.Mutex{}
	similarUsersRatings := make(map[int]float32)

	for otherUserId, ratings := range userRatings {
		if otherUserId == userId {
			continue
		}

		wg.Add(1)
		go func(otherUserId int, ratings map[int]float32) {
			defer wg.Done()
			localRatings := make(map[int]float32)
			for movieId, rating := range ratings {
				if _, rated := userRatings[userId][movieId]; !rated {
					localRatings[movieId] += rating
				}
			}

			mu.Lock()
			for movieId, rating := range localRatings {
				similarUsersRatings[movieId] += rating
			}
			mu.Unlock()
		}(otherUserId, ratings)
	}

	wg.Wait()

	// Sort movies by total rating
	recommendations := make([]int, 0, len(similarUsersRatings))
	for movieId := range similarUsersRatings {
		recommendations = append(recommendations, movieId)
	}

	return recommendations
}
