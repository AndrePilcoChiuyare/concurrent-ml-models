package cf

import (
	"encoding/csv"
	"os"
	"strconv"
)

// Data structures
type UserRatings map[int]map[int]float32 // userId -> movieId -> rating

// Load CSV file into a map
func LoadCSV(filename string) (UserRatings, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	userRatings := make(UserRatings)
	for _, record := range records[1:] { // Skip header
		userId, _ := strconv.Atoi(record[0])
		movieId, _ := strconv.Atoi(record[1])
		rating, _ := strconv.ParseFloat(record[2], 32)

		if _, exists := userRatings[userId]; !exists {
			userRatings[userId] = make(map[int]float32)
		}
		userRatings[userId][movieId] = float32(rating)
	}
	return userRatings, nil
}

// Recommend movies based on user ratings
func RecommendMovies(userId int, userRatings UserRatings) []int {
	// Collect all movies rated by similar users
	similarUsersRatings := make(map[int]float32)
	for otherUserId, ratings := range userRatings {
		if otherUserId == userId {
			continue
		}
		for movieId, rating := range ratings {
			if _, rated := userRatings[userId][movieId]; !rated {
				similarUsersRatings[movieId] += rating
			}
		}
	}

	// Sort movies by total rating
	recommendations := make([]int, 0, len(similarUsersRatings))
	for movieId := range similarUsersRatings {
		recommendations = append(recommendations, movieId)
	}

	return recommendations
}
