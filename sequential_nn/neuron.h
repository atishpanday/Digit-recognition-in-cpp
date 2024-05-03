#ifndef NEURON
#define NEURON

#include "utils.h"

const double alpha = 0.01;

class neuron {
	int num_weights;
	double value;
	std::vector<double> weights;
	
	public:
	neuron(int K, double stddev) {
		std::random_device rd;  // Obtain a random seed from the OS entropy device
    		std::mt19937 gen(rd()); // Seed the generator
		
    		// Create a uniform distribution between 0 and 1
    		std::normal_distribution<double> dist(0.0, stddev);
    		
    		num_weights = K;
    		weights.reserve(K);
		for(int i = 0; i < K; i++) {
			weights.push_back(dist(gen));
		}
	}
	
	void calculate_value(std::vector<double>& x, const std::string& activation) {
		value = 0;
		for(int i = 0; i < num_weights; i++) {
			value += weights[i] * x[i];
		}
		
		if(activation == "softmax" || activation == "linear") {
		}
		else {
			activate(activation);
		}
		
	}
	
	void activate(const std::string& activation) {
		if(activation == "sigmoid") {
			value = sigmoid(value);
		}
		else {
			value = relu(value);
		}
	}
	
	void activate_softmax(std::vector<double>& z) {
		value = softmax(value, z);
	}
	
	void update_weights(std::vector<double>& x, double t, std::string& activation) {
		for(int i = 0; i < num_weights; i++) {
			if(weights[i] < 0) {
				weights[i] -= alpha * get_gradient(t, x[i], activation);
			}
			else {
				weights[i] += alpha * get_gradient(t, x[i], activation);
			}
		}
	}
	
	double get_gradient(double t, double x, std::string& activation) {
		if(activation == "softmax") {
			return (value - t) * x;
		}
		else if(activation == "sigmoid") {
			return (value - t) * value * (1 - value) * x;
		}
		else {
			return x > 0 ? (t * x / value) : 0;
		}
	}
	
	double get_value() {
		return value;
	}
};

#endif
