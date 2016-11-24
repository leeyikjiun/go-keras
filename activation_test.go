package keras

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRelu(t *testing.T) {
	assert.Equal(t, 0.0, relu(-0.1))
	assert.Equal(t, 0.0, relu(0.0))
	assert.Equal(t, 0.1, relu(0.1))
}
