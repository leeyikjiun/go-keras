package keras

import "math"

type Activation func(float64) float64

func relu(x float64) float64 {
	return math.Max(0.0, x)
}
