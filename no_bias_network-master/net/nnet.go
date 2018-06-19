package nbnet

import (
	"fmt"
	"math"
	"os"
	"path"

	"gonum.org/v1/gonum/mat"
)

//
// three-layer network without bias
//

type Network struct {
	// neuron counts for each layer
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}

//
// accessors
//

func (net *Network) InputCount() int {
	return net.inputs
}

func (net *Network) HiddenCount() int {
	return net.hiddens
}

func (net *Network) OutputCount() int {
	return net.outputs
}

// creates network and initializes weights with random values
func CreateNetwork(input, hidden, output int, learningRate float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: learningRate,
	}

	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.hiddens*net.inputs, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.outputs*net.hiddens, float64(net.hiddens)))
	return
}

func (net *Network) Train(inputData []float64, targetData []float64) {
	// forward feed process
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := applyFunction(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := applyFunction(sigmoid, finalInputs)

	// calculate error
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := subtract(targets, finalOutputs)
	hiddenErrors := dot(net.outputWeights.T(), outputErrors)

	// backprop
	net.outputWeights = add(net.outputWeights, scale(net.learningRate,
		dot(multiply(outputErrors, sigmoidComposite(finalOutputs)),
			hiddenOutputs.T()))).(*mat.Dense)
	net.hiddenWeights = add(net.hiddenWeights, scale(net.learningRate,
		dot(multiply(hiddenErrors, sigmoidComposite(hiddenOutputs)),
			inputs.T()))).(*mat.Dense)
}

func (net *Network) Predict(inputData []float64) mat.Matrix {
	// forward propagation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := applyFunction(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := applyFunction(sigmoid, finalInputs)
	return finalOutputs
}

//
// save and loading functions
//

func Save(net Network) {
	folderName := "modelweights"
	err := os.MkdirAll(folderName, os.ModePerm)

	if err != nil {
		fmt.Println("unable to create weights directory ", err)
	}

	hidden, err := os.Create(path.Join(folderName, "hweights.nbmod"))
	defer hidden.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(hidden)
	} else {
		fmt.Println("unable to create hweights file ", err)
	}

	output, err := os.Create(path.Join(folderName, "oweights.nbmod"))
	defer output.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(output)
	}
}

func Load(net *Network) {
	hidden, err := os.Open("modelweights/hweights.nbmod")
	defer hidden.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(hidden)
	}

	output, err := os.Open("modelweights/oweights.nbmod")
	defer output.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(output)
	}
	return
}

//
// activation functions
//

func sigmoid(row, column int, z float64) float64 { // needs to match the signature of applyFunction
	return 1.0 / (1 + math.Exp(-1*z))
}

// composite of sigmoid function for backprop - equal to sigmoid(1 - sigmod)
func sigmoidComposite(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	onesArray := make([]float64, rows)
	for i := range onesArray {
		onesArray[i] = 1
	}

	ones := mat.NewDense(rows, 1, onesArray)
	return multiply(m, subtract(ones, m))
}
