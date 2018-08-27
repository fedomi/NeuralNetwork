#pragma once
#include <math.h>
#include <vector>
#include "Neuron.h"
#include <random>
#include<limits>
using namespace std;

class NeuralNetwork
{

public:
	NeuralNetwork();
	~NeuralNetwork();

	void Train(vector<float> input, vector<float> target);
	void Test(vector<float> input, vector<float> target);

	void CheckLayerDimensions();
	float MSE();

	// These methods must be private. They're here for testing
	float PropagateValue(float input);	
	void UpdateWeights(float target);
private:
	float alpha = 0.5;
	float MSE_Threshold = 0.25;
	float iterations_limit = 1000;

	float NI = 1;	// Number of input neurons
	float NO = 1;	// Number of output neurons
	float NHL = 1;	// Number of hidden layers
	float NH = 3;	// Number of neurons per hidden layer

	vector<float> errors;

	// TODO: New way of organizing neurons in layers, so that any grid of neurons/layers is possible
	vector<Neuron> inputLayer;
	vector<Neuron> outputLayer;
	vector<Neuron> hiddenLayer;
	
	void CreateLayers();
	//float PropagateValue(float input);
	//void UpdateWeights();
};

