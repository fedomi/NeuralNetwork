#include <iostream>
#include <vector>
#include <math.h>
#include "Neuron.h"
#include "NeuralNetwork.h"
using namespace std;

// Tests
void TestNeuronOutput();
void TestNetworkCreation();
void TestInputPropagation();
void TestWeightUpdate();
void TestTrain();

int main() {
	int c;

	TestTrain();
	cin >> c;
	return 0;
}


// PASSED
void TestNeuronOutput() {
	vector<float> input, weights;
	vector<Neuron> next;

	input.push_back(0.25f);
	weights.push_back(0.25f);

	Neuron n(0.1f, weights, -1, next);

	//cout << 1/ (1 + exp(1.0625)) << endl;
	cout << "Output: " << n.Output(input) << endl;
}

// PASSED
void TestNetworkCreation() {

	NeuralNetwork nn;

	nn.CheckLayerDimensions();

}

// PASSED
void TestInputPropagation() {

	NeuralNetwork nn;
	 
	cout << nn.PropagateValue(1) << endl;

}

// PASSED
void TestWeightUpdate() {
	NeuralNetwork nn;

	cout << nn.PropagateValue(1) << endl;
	nn.UpdateWeights();

}


void TestTrain() {
	NeuralNetwork nn;
	vector<float> input;
	vector<float> target;

	input.push_back(1);
	target.push_back(1);

	input.push_back(2);
	target.push_back(2);

	nn.Train(input, target);

}