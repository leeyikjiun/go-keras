package keras

type Layer interface {
	GetWeights() ([][]float64, []float64)
	SetWeights([][]float64, []float64)
	getOutputDim() int
	getInputDim() int
	setInputDim(int)
	activation(float64) float64
}

type Dense struct {
	OutputDim  int
	Activation Activation
	Weights    [][]float64
	Bias       []float64
	InputDim   int
}

func (d Dense) GetWeights() ([][]float64, []float64) {
	return d.Weights, d.Bias
}

func (d *Dense) SetWeights(weights [][]float64, bias []float64) {
	d.Weights = weights
	d.Bias = bias
}

func (d Dense) getOutputDim() int {
	return d.OutputDim
}

func (d Dense) getInputDim() int {
	return d.InputDim
}

func (d *Dense) setInputDim(dim int) {
	d.InputDim = dim
}

func (d Dense) activation(x float64) float64 {
	return d.Activation(x)
}
