#pragma once
#include <vector>
#include <iostream>
using namespace std;
class Neuron
{
public:
	Neuron(float learning_rate, vector<float> weights, float threshold, vector<Neuron> next);
	Neuron(float learning_rate, vector<float> weights, float threshold);
	~Neuron();

	float Output(vector<float> in);
	float Gradient(float d);
	float Gradient(vector<float> gk);

	void UpdateWeights(float delta);

private:
	vector<float> w;	// Weights
	vector<float> input;
	float output;

	float t;		// Threshold
	float alpha;	// Learning rate
	float delta;	// Error gradient
	vector<Neuron> next_layer;

	float Sigmoid(float x);
	float dSigmoid(float x);
};

