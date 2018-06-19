package nbnet

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// helper functions to reduce code clutter, all underlying mat operation functions require
// both the arrays have the same shape except when computing the dot product

// computes the dot product
func dot(m, n mat.Matrix) mat.Matrix { //
	// needs the rows of m and columns of n to get the shape of the final matrix
	rows, _ := m.Dims()
	_, columns := n.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.Product(m, n)
	return result
}

// applies a funciton to a matrix
func applyFunction(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix { //
	rows, columns := m.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.Apply(fn, m)
	return result
}

// scales a matrix by a factor
func scale(scalar float64, m mat.Matrix) mat.Matrix { //
	rows, columns := m.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.Scale(scalar, m)
	return result
}

// multiplies two matrices together
func multiply(m, n mat.Matrix) mat.Matrix { //
	rows, columns := m.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.MulElem(m, n)
	return result
}

// adds two matrices together
func add(m, n mat.Matrix) mat.Matrix { //
	rows, columns := m.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.Add(m, n)
	return result
}

// adds a scalar to all indices in a matrix
func addScalar(scalar float64, m mat.Matrix) mat.Matrix { //
	rows, columns := m.Dims()
	summand := make([]float64, rows*columns)
	for i := range summand {
		summand[i] = scalar
	}

	result := mat.NewDense(rows, columns, summand)
	result.Add(result, m)
	return result
}

// subtracts matrix n from m
func subtract(m, n mat.Matrix) mat.Matrix { //
	rows, columns := m.Dims()
	result := mat.NewDense(rows, columns, nil)
	result.Sub(m, n)
	return result
}

// generates a random array for initialization, uses the size of the layer to generate values
func randomArray(arraySize int, layerSize float64) (data []float64) { //
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(layerSize),
		Max: 1 / math.Sqrt(layerSize),
	}

	data = make([]float64, arraySize)
	for i := range data {
		data[i] = dist.Rand()
	}
	return
}

// prints a formatted matrix
func printMatrix(m mat.Matrix) { //
	formatted := mat.Formatted(m, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v", formatted)
}
