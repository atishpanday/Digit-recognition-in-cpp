#ifndef LAYER
#define LAYER

#include "neuron.h"

class layer {
	int num_neurons;
	std::vector<neuron> neurons;
	std::vector<double> values;
	std::string activation;
	double loss;
	
	public:
	layer(int K, int N, const std::string& act) {
		num_neurons = N;
		activation = act;
		neurons.reserve(N);
		values.reserve(N);
		double sd;
		if(act == "relu") {
			sd = sqrt(2.0 / K);
		}
		else {
			sd = sqrt(2.0 / (K + N));
		}
		for(int i = 0; i < N; i++) {
			neurons.emplace_back(K, sd);
			values.push_back(0);
		}
	}
	
	void calculate_values(std::vector<double>& x) {
		for(int i = 0; i < num_neurons; i++) {
			neurons[i].calculate_value(x, activation);
			values[i] = neurons[i].get_value();
		}
		
		if(activation == "softmax") {
			for(int i = 0; i < num_neurons; i++) {
				neurons[i].activate_softmax(values);
			}
			for(int i = 0; i < num_neurons; i++) {
				values[i] = neurons[i].get_value();
			}
		}
	}
	
	void update_weights(std::vector<double>& x, std::vector<double>& t) {
		for(int i = 0; i < num_neurons; i++) {
			neurons[i].update_weights(x, t[i], activation);
		}
		calculate_values(x);
		calculate_loss(t);
	}

	// Cross entropy loss function
	void calculate_loss(std::vector<double>& t) {
		double y;
		loss = 0;
		for(int i = 0; i < num_neurons; i++) {
			y = neurons[i].get_value();
			loss -= t[i] * log(y);
		}
	}

	int get_num_neurons() {
		return num_neurons;
	}

	std::vector<double>& get_values() {
		return values;
	}
	
	double get_loss() {
		return loss;
	}
};

#endif
