package keras

type Model interface {
	Predict([]float64)
}

type Sequential struct {
	layers []Layer
}

func (s Sequential) Predict(x []float64) []float64 {
	for _, layer := range s.layers {
		w, b := layer.GetWeights()
		y := make([]float64, layer.getOutputDim())
		for j := 0; j < layer.getOutputDim(); j++ {
			y[j] = b[j]
			for i := 0; i < layer.getInputDim(); i++ {
				y[j] += x[i] * w[i][j]
			}
			y[j] = layer.activation(y[j])
		}
		x = y
	}
	return x
}
