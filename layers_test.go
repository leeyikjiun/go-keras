package keras

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetWeights(t *testing.T) {
	layer := Dense{}
	layer.Weights = [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	layer.Bias = []float64{1, 2, 3}

	weights, bias := layer.GetWeights()
	assert.Equal(t, layer.Weights, weights)
	assert.Equal(t, layer.Bias, bias)
}

func TestSetWeights(t *testing.T) {
	layer := Dense{}
	weights := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	bias := []float64{1, 2, 3}

	layer.SetWeights(weights, bias)
	assert.Equal(t, weights, layer.Weights)
	assert.Equal(t, bias, layer.Bias)
}
