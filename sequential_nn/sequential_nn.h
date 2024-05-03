#ifndef SEQUENTIAL_NN
#define SEQUENTIAL_NN

#include "layer.h"

class sequential_nn {
	std::vector<layer> layers;
	int num_layers;
	
	void forward(std::vector<double>& image) {
		layers[0].calculate_values(image);
		for(int i = 1; i < num_layers; i++) {
			layers[i].calculate_values(layers[i - 1].get_values());
		}
	}

	void backpropagation(std::vector<double>& image, std::vector<double>& label) {
		layers[num_layers - 1].update_weights(layers[num_layers - 2].get_values(), label);
		for(int i = (num_layers - 2); i > 0; i++) {
			layers[i].update_weights(layers[i - 1].get_values(), layers[i + 1].get_values());
		}
		layers[0].update_weights(image, layers[1].get_values());
	}

	public:
	sequential_nn(std::vector<layer>& l) {
		num_layers = l.size();
		for(layer ll:l) {
			layers.push_back(ll);
		}
	}
		
	void fit(std::vector<double>& image, std::vector<double>& label) {
		forward(image);
		backpropagation(image, label);
	}

	int predict(std::vector<double>& image) {
		forward(image);
		int pred = layers[num_layers - 1].get_values()[0];
		int max_ind;
		for(int i = 0; i < layers[num_layers - 1].get_num_neurons(); i++) {
			if(pred < layers[num_layers - 1].get_values()[i]) {
				pred = layers[num_layers - 1].get_values()[i];
				max_ind = i;
			}
		}
		return max_ind;
	}
	
	double get_loss() {
		return layers[num_layers - 1].get_loss();
	}
};

#endif
