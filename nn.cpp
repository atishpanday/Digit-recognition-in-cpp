// Neural network implementation

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

#define alpha 0.01
const int epochs = 100;
const int training_size = 50000;
const int image_size = 28 * 28;
const int batch_size = 500;

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double relu(double x) {
	return x > 0 ? x : 0;
}

double softmax(double x, std::vector<double>& z) {
	double exp_sum = 0;
	for(double zi:z) {
		exp_sum += exp(-zi);
	}
	return exp(-x) / exp_sum;
}

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
	int num_weights;
	double value;
	std::vector<double> weights;
	// std::vector<double> bias;
	
	public:
	neuron(int K) {
		std::random_device rd;  // Obtain a random seed from the OS entropy device
    		std::mt19937 gen(rd()); // Seed the generator

    		// Create a uniform distribution between 0 and 1
    		std::uniform_real_distribution<double> dist(0.0, 1.0);
    		
    		num_weights = K;
    		weights.reserve(K);
		for(int i = 0; i < K; i++) {
			weights.push_back(dist(gen));
			// bias.push_back(dist(gen));
		}
		value = 0;
	}
	
	void calculate_value(std::vector<double>& x, const std::string& activation) {
		for(int i = 0; i < num_weights; i++) {
			value += weights[i] * x[i]; // + bias[i];
		}
		
		if(activation == "softmax") {
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
	
	void update_weights(std::vector<double>& x, double t) {
		for(int i = 0; i < num_weights; i++) {
			weights[i] -= alpha * (value - t) * value * (1 - value) * x[i];
		}
	}
	
	double get_value() {
		return value;
	}
	
	// ~neuron();
};

class layer {
	int num_neurons;
	std::vector<neuron> neurons;
	std::vector<double> values;
	double loss;
	std::string activation;
	
	public:
	layer(int K, int N, const std::string& act) {
		num_neurons = N;
		activation = act;
		neurons.reserve(N);
		values.reserve(N);
		for(int i = 0; i < N; i++) {
			neurons.emplace_back(K);
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
			neurons[i].update_weights(x, t[i]);
		}
		calculate_loss(t);
	}

	void calculate_loss(std::vector<double>& t) {
		double y;
		loss = 0;
		for(int i = 0; i < num_neurons; i++) {
			y = neurons[i].get_value();
			loss += (t[i] - y) * (t[i] - y);
		}
		loss /= 2 * num_neurons;
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
	// ~layer();
};

double predict(std::vector<double>& image, layer& layer1, layer& layer2, layer& layer3, layer& output_layer) {
	layer1.calculate_values(image);
	layer2.calculate_values(layer1.get_values());
	layer3.calculate_values(layer2.get_values());
	output_layer.calculate_values(layer3.get_values());
	
	return output_layer.get_values()[0] * 10;
}

int main() {
	std::vector<double> image(image_size);
	double label;	
	std::string train_image_file = "train-images-idx3-ubyte";
	std::string train_label_file = "train-labels-idx1-ubyte";
	
	std::ifstream tif(train_image_file, std::ios::binary);
	std::ifstream tlf(train_label_file, std::ios::binary);
	
	// Initializing the layers
	layer layer1 = layer(image_size, 128, "relu");
	layer layer2 = layer(128, 64, "relu");
	layer layer3 = layer(64, 10, "softmax");
	layer output_layer = layer(10, 1, "sigmoid");
	
	double avg_loss;
	
	std::ofstream loss("loss.txt");
	
	int ep = 0;
	
	while(ep < epochs) {
		// Assuming image has stored 1 image data from the MNIST dataset
		avg_loss = 0;
		int i = ep * batch_size;
		while(i < (ep + 1) * batch_size) {
			read_mnist_train_image(tif, image, i);
			label = static_cast<double>(read_mnist_train_label(tlf, i));
			
			layer1.calculate_values(image);
			layer2.calculate_values(layer1.get_values());
			layer3.calculate_values(layer2.get_values());
			output_layer.calculate_values(layer3.get_values());
			
			// predictions[i] = output_layer.get_values()[0];
			std::vector<double> temp_label = {label / 10};
			output_layer.update_weights(layer3.get_values(), temp_label);
			layer3.update_weights(layer2.get_values(), output_layer.get_values());
			layer2.update_weights(layer1.get_values(), layer3.get_values());
			layer1.update_weights(image, layer2.get_values());
			
			avg_loss += output_layer.get_loss();
			
			i++;
		}
		
		avg_loss /= batch_size;
		
		loss << avg_loss << std::endl;
		std::cout << "Epoch: " << ep << "\n";
		ep++;
	}
	
	loss.close();
	
	read_mnist_train_image(tif, image, training_size + 1000);
	label = static_cast<double>(read_mnist_train_label(tlf, training_size + 1000));
	
	double prediction = std::round(predict(image, layer1, layer2, layer3, output_layer));
	
	std::cout << "\nPredicted value: " << prediction << "\nTrue value: " << label << "\n";
	
	tif.close();
	tlf.close();
		
	
	return 0;
}
