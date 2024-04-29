// Neural network implementation

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

#define alpha 0.001
#define epochs 10
#define training_size 10000
#define image_size 28 * 28

double sigmoid(double x) {
	return (double) 1 / (1 + exp(-x));
}

/*double relu(double x) {
	return x > 0 ? x : 0;
}*/

void read_mnist_train_image(std::ifstream& train_file, std::vector<double>& image, int i) {
	train_file.seekg(4 * sizeof(int) + i * image_size, std::ios::beg);
	std::vector<unsigned char> temp(image_size);
	train_file.read(reinterpret_cast<char*>(temp.data()), image_size);
	
	for(int j = 0; j < image_size; j++) {
		image[j] = static_cast<double>(temp[j]) / 255.0;
	}
}

unsigned char read_mnist_train_label(std::ifstream& label_file, int i) {
	label_file.seekg(2 * sizeof(int) + i, std::ios::beg);
	unsigned char label;
	label_file.read(reinterpret_cast<char*>(&label), 1);
	
	return label;
}

class neuron {
	double value;
	std::vector<double> weights;
	// std::vector<double> bias;
	
	public:
	neuron(int K) {
		std::random_device rd;  // Obtain a random seed from the OS entropy device
    		std::mt19937 gen(rd()); // Seed the generator

    		// Create a uniform distribution between 0 and 1
    		std::uniform_real_distribution<double> dist(0.0, 1.0);
    		
    		weights.resize(K);
		for(int i = 0; i < K; i++) {
			weights[i] = dist(gen);
			// bias.push_back(dist(gen));
		}
		value = 0;
	}
	
	void calculate_value(std::vector<double>& x, int K) {
		for(int i = 0; i < K; i++) {
			value += weights[i] * x[i]; // + bian[i];
		}
		
		value = sigmoid(value);
	}
	
	void update_weights(std::vector<double>& x, double t, int K) {
		for(int i = 0; i < K; i++) {
			weights[i] -= alpha * (value - t) * value * (1 - value) * x[i];
		}
	}
	
	double get_weight(int i) {
		return weights[i];
	}
	
	//double get_bias(int i) {
	//	return bias[i];
	//}
	
	double get_value() {
		return value;
	}
	
	// ~neuron();
};

class layer {
	std::vector<neuron> neurons;
	std::vector<double> values;
	double loss;
	
	public:
	layer(int K, int N) {
		for(int i = 0; i < N; i++) {
			neurons.push_back(neuron(K));
		}
	}
	
	void calculate_values(std::vector<double>& x, int K) {
		values.resize(K);
		int i = 0;
		for(neuron n:neurons) {
			n.calculate_value(x, K);
			values[i] = n.get_value();
			i++;
		}
	}
	
	double get_loss() {
		return loss;
	}
	
	std::vector<double>& get_values() {
		return values;
	}
	
	void calculate_loss(std::vector<double>& t) {
		double y;
		int i = 0;
		for(neuron n:neurons) {
			y = n.get_value();
			loss += (t[i] - y) * (t[i] - y);
			i++;
		}
		loss /= 2;
	}
	
	void update_weights(std::vector<double>& x, std::vector<double>& t, int K) {
		calculate_loss(t);
		int i = 0;
		for(neuron n:neurons) {
			n.update_weights(x, t[i], K);
			i++;
		}
	}
	
	// ~layer();
};

int main() {
	std::vector<double> image(image_size);
	double label;	
	std::string train_image_file = "train-images-idx3-ubyte";
	std::string train_label_file = "train-labels-idx1-ubyte";
	
	std::ifstream tif(train_image_file, std::ios::binary);
	std::ifstream tlf(train_label_file, std::ios::binary);
	
	// Initializing the layers
	layer layer1 = layer(image_size, 128);
	layer layer2 = layer(128, 64);
	layer layer3 = layer(64, 10);
	layer output_layer = layer(10, 1);
	
	double avg_loss;
	
	std::ofstream loss("loss.txt");
	
	int ep = 0;
	
	while(ep < epochs) {
		// Assuming image has stored 1 image data from the MNIST dataset
		avg_loss = 0;
		int i = 0;
		while(i < training_size) {
			read_mnist_train_image(tif, image, i);
			label = static_cast<double>(read_mnist_train_label(tlf, i));
			
			layer1.calculate_values(image, image.size());
			layer2.calculate_values(layer1.get_values(), layer1.get_values().size());
			layer3.calculate_values(layer2.get_values(), layer2.get_values().size());
			output_layer.calculate_values(layer3.get_values(), layer3.get_values().size());
			
			// predictions[i] = output_layer.get_values()[0];
			std::vector<double> temp_label = {label};
			output_layer.update_weights(layer3.get_values(), temp_label, layer3.get_values().size());
			layer3.update_weights(layer2.get_values(), output_layer.get_values(), layer2.get_values().size());
			layer2.update_weights(layer1.get_values(), layer3.get_values(), layer1.get_values().size());
			layer1.update_weights(image, layer2.get_values(), image.size());
			
			avg_loss += output_layer.get_loss();
			
			i++;
		}
		
		avg_loss /= training_size;
		
		loss << avg_loss << std::endl;
		std::cout << "\nEpoch: " << ep;
		ep++;
	}
	
	loss.close();
		
	
	return 0;
}
