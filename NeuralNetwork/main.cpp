#include <iostream>
#include <vector>
#include <math.h>
#include "Neuron.h"
#include "NeuralNetwork.h"
#include <fstream>
#include <string>
#include <utility>
using namespace std;

// Tests
void TestNeuronOutput();
void TestNetworkCreation();
void TestInputPropagation();
void TestWeightUpdate();
void TestTrain();
void TestRead();
pair<vector<float>, vector<float>> ReadData(string file);



int main() {
	int c;

	TestRead();
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
	float output = nn.PropagateValue(1);

	cout << output << endl;
	nn.UpdateWeights(1);

}

// PASSED
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


void TestRead() {
	vector<float> TrainIn, TrainTarget, TestIn, TestTarget;
	pair<vector<float>, vector<float>> r1, r2;
	r1 = ReadData("sincTrain25.dt");
	r2 = ReadData("sincValidate10.dt");
	TrainIn = r1.first;
	TrainTarget = r1.second;
	TestIn = r2.first;
	TestTarget = r2.second;

}

pair<vector<float>, vector<float>> ReadData(string file) {

	ifstream input(file); //put your program together with thsi file in the same folder.
	vector<float> in;
	vector<float> target;

	if (input.is_open()) {

		while (!input.eof()) {
			string numbers;
			int data;
			getline(input, numbers); //read number
			string::size_type sz;     // alias of size_t

			float i = stof(numbers, &sz);
			float t = stof(numbers.substr(sz));
			in.push_back(i);
			target.push_back(t);
		}

	}

	pair <vector<float>, vector<float>> result(in, target);
	//result.first = input;
	//result.second = target;
	return result;
}