#include "Neuron.h"



Neuron::Neuron(float learning_rate, vector<float> weights, float threshold, vector<Neuron> next)
{
	w = weights;
	t = threshold;
	alpha = learning_rate;
	next_layer = next;
}

Neuron::Neuron(float learning_rate, vector<float> weights, float threshold)
{
	w = weights;
	t = threshold;
	alpha = learning_rate;
}


Neuron::~Neuron()
{
}



float Neuron::Sigmoid(float x)
{
	return 1.0 / (1.0 + exp(-x));

}

float Neuron::dSigmoid(float x)
{
	float sx = Sigmoid(x);
	return sx * (1 - sx);

}


float Neuron::Output(vector<float> in) {
	input = in;

	float res = 0;
	auto n = in.size();
	if (w.size() == n) {		// If input and weights vectors have the same length
		for (int i = 0; i < n; i++) {
			res += w[i] * in[i];
		}
	}

	res -= t;
	output = Sigmoid(res);
	return output;
}


// Gradient function for output layer neurons
float Neuron::Gradient(float d) 
{
	float res = output * (1 - output)*(d - output);

	delta = res;
	return res;

}

// Gradient function for hidden layer neurons
float Neuron::Gradient(vector<float> gk) 
{	
	float res = 0;

	for (int i = 0; i < gk.size(); i++)
	{
		res += w[i] * gk[i];
	}

	res *= output * (1 - output);

	delta = res;
	return res;
}


// Updates the weights of this neuron
void Neuron::UpdateWeights(float delta) {

	for (int i = 0; i < w.size(); i++) {
		w[i] += alpha * input[i] * delta;
		//cout << "New weights: " << w[i] << endl;	// Debug
	}

}