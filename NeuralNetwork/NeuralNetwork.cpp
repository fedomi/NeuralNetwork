#include "NeuralNetwork.h"



NeuralNetwork::NeuralNetwork()
{
	CreateLayers();
}


NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::CreateLayers() {
	// TODO: Change weight initialization to a binomial distribution
	
	default_random_engine generator;
	binomial_distribution<int> distribution(1000, 0.5);
	int number = distribution(generator);

	vector<float> default_weights_in, default_weights_hid, default_weights_out;	// For testing purposes
	default_weights_in.push_back(distribution(generator)/1000);
	
	default_weights_hid.push_back(distribution(generator) / 1000);

	default_weights_out.push_back(distribution(generator) / 1000);
	default_weights_out.push_back(distribution(generator) / 1000);
	default_weights_out.push_back(distribution(generator) / 1000);

	
	for (int i = 0; i < NI; i++) {
		Neuron n(alpha, default_weights_in, -1);
		inputLayer.push_back(n);
	}

	for (int i = 0; i < NH; i++) {
		Neuron n(alpha, default_weights_hid, -1);
		hiddenLayer.push_back(n);
	}

	for (int i = 0; i < NO; i++) {
		Neuron n(alpha, default_weights_out, -1);
		outputLayer.push_back(n);
	}
}

void NeuralNetwork::Train(vector<float> input, vector<float> target) {
	/* TODO:
		1. For every input, propagate values forward inside the network
		2. Get the error based on the output and the target value
		3. Backpropagate the error and update weights
	*/
	float out, mse = numeric_limits<float>::max(), num_iterations = 0;
	while (mse > MSE_Threshold || num_iterations < iterations_limit) {
		for (int i = 0; i < input.size(); i++) {
			out = PropagateValue(input[i]);
			errors.push_back(out - target[i]);
			UpdateWeights(target[i]);
		}
		mse = MSE();
		cout << "Mean Squared Error: " << mse << endl;
		num_iterations++;
	}
	
}

void NeuralNetwork::Test(vector<float> input, vector<float> target) {
	/* TODO:
	1. For every input, propagate values forward inside the network
	2. Compute precision of the outputs
	*/

	float out;
	for (int i = 0; i < input.size(); i++) {
		out = PropagateValue(input[i]);
		errors.push_back(out - target[i]);
		//UpdateWeights();
	}
	cout << "Mean Squared Error: " << MSE() << endl;
}


float NeuralNetwork::PropagateValue(float value) {
	// TODO: Add support for multiple neurons in input and output layers, as well as more hidden layers
	vector<float> vin, vhid, vout;		// These are the values passed as entries in each of the three layers
	vin.push_back(value);
	
	// Input layer propagation
	float oil = inputLayer[0].Output(vin);
	// Hidden layer propagation
	vhid.push_back(oil);
	for (int i = 0; i < hiddenLayer.size(); i++) {
		vout.push_back(hiddenLayer[i].Output(vhid));
	}
	// Output layer propagation
	float result = outputLayer[0].Output(vout);

	return result;
}

void NeuralNetwork::UpdateWeights(float target){
	vector<float> vin, vhid, vout;		// These are the values passed as entries in each of the three layers
	
	
	float gk = outputLayer[0].Gradient(target);
	outputLayer[0].UpdateWeights(gk);
	//cout << "Error gradient at output neuron: " << gk << endl;
										
	vout.push_back(gk);
	float gj;
	for (int i = 0; i < hiddenLayer.size(); i++) {
		gj = hiddenLayer[i].Gradient(vout);
		hiddenLayer[i].UpdateWeights(gj);
		//cout << "Error gradient at hidden neuron: " << gj << endl;
		vhid.push_back(gj);
	}
	
	float gi = inputLayer[0].Gradient(vhid);
	inputLayer[0].UpdateWeights(gi);
	
}


void NeuralNetwork::CheckLayerDimensions() {

	cout << "Input layer neurons: " << inputLayer.size() << endl;
	cout << "Hidden layer neurons: " << hiddenLayer.size() << endl;
	cout << "Output layer neurons: " << outputLayer.size() << endl;


}

float NeuralNetwork::MSE() {
	int n = errors.size();
	float mse = 0;

	for (int i = 0; i < n; i++) {
		mse += errors[i]*errors[i];
	}

	mse /= n;

	return mse;
}