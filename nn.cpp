// Neural network implementation

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

const double alpha = 0.01;
const int epochs = 5;
const int total_size = 60000;
const int training_size = 50000;
const int validation_size = total_size - training_size;
const int image_size = 28 * 28;
//const int batch_size = 500;
bool verbose = 0;

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double relu(double x) {
	return x > 0 ? x : 0;
}

double softmax(double x, std::vector<double>& z) {
	double exp_sum = 0;
	for(int i = 0; i < z.size(); i++) {
		exp_sum += exp(z[i]);
	}
	return exp(x) / exp_sum;
}

void read_mnist_train_image(std::ifstream& train_file, std::vector<double>& image, int i) {
	train_file.seekg(4 * sizeof(int) + i * image_size, std::ios::beg);
	std::vector<unsigned char> temp(image_size);
	train_file.read(reinterpret_cast<char*>(temp.data()), image_size);
	
	for(int j = 0; j < image_size; j++) {
		image[j] = static_cast<double>(temp[j]) / 255.0;
	}
}

int read_mnist_train_label(std::ifstream& label_file, int i) {
	label_file.seekg(2 * sizeof(int) + i, std::ios::beg);
	unsigned char label;
	label_file.read(reinterpret_cast<char*>(&label), 1);
	
	return static_cast<int>(label);
}

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
			if(verbose) std::cout << "\nSigmoid: " << value;
		}
		else {
			value = relu(value);
			if(verbose) std::cout << "\nRelu: " << value;
		}
	}
	
	void activate_softmax(std::vector<double>& z) {
		value = softmax(value, z);
		if(verbose) std::cout << "\nSoftmax: " << value;
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
		else {
			return x > 0 ? (t * x / value) : 0;
		}
	}
	
	double get_value() {
		return value;
	}
};

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
		if(verbose) std::cout << "\nSD: " << sd;
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

void forward(std::vector<double>& image, layer& layer1, layer& layer2, layer& output_layer) {
	layer1.calculate_values(image);
	layer2.calculate_values(layer1.get_values());
	output_layer.calculate_values(layer2.get_values());
}

void backpropagation(std::vector<double>& image, std::vector<double>& label, layer& layer1, layer& layer2, layer& output_layer) {
	output_layer.update_weights(layer2.get_values(), label);
	layer2.update_weights(layer1.get_values(), output_layer.get_values());
	layer1.update_weights(image, layer2.get_values());
}

int predict(std::vector<double>& image, layer& layer1, layer& layer2, layer& output_layer) {
	forward(image, layer1, layer2, output_layer);
	int pred = output_layer.get_values()[0];
	int max_ind;
	for(int i = 0; i < output_layer.get_num_neurons(); i++) {
		if(pred < output_layer.get_values()[i]) {
			pred = output_layer.get_values()[i];
			max_ind = i;
		}
	}
	return max_ind;
}

int main() {
	std::vector<double> image(image_size);
	std::vector<double> label(10, 0.0);
	int ind;
	
	std::string train_image_file = "train-images-idx3-ubyte";
	std::string train_label_file = "train-labels-idx1-ubyte";
	std::ifstream tif(train_image_file, std::ios::binary);
	std::ifstream tlf(train_label_file, std::ios::binary);
	
	int layer1_neurons = 128;
	int layer2_neurons = 64;
	int output_layer_neurons = 10;
	
	// Initializing the layers
	layer layer1 = layer(image_size, layer1_neurons, "relu");
	layer layer2 = layer(layer1_neurons, layer2_neurons, "relu");
	layer output_layer = layer(layer2_neurons, output_layer_neurons, "softmax");
	
	//if(false) {
	double avg_loss;
	
	std::cout << "---------------Training---------------";
	
	std::ofstream loss("loss.txt");
	
	int ep = 0;
	
	while(ep < epochs) {
		avg_loss = 0;
		int tr = 0;
		while(tr < training_size) {
			read_mnist_train_image(tif, image, tr);
			ind = read_mnist_train_label(tlf, tr);
			label[ind] = 1.0;
			
			forward(image, layer1, layer2, output_layer);
			backpropagation(image, label, layer1, layer2, output_layer);
			
			label[ind] = 0.0;
			avg_loss += output_layer.get_loss();
			tr++;
		}
		
		avg_loss /= training_size;
		
		loss << avg_loss << std::endl;
		std::cout << "\nEpoch: " << ep;
		ep++;
	}
	
	
	loss.close();
	
	std::cout << "\n---------------Validating---------------";
	int prediction;
	int te = training_size;
	double accuracy = 0;
	while(te < total_size) {
		read_mnist_train_image(tif, image, te);
		ind = read_mnist_train_label(tlf, te);
		prediction = predict(image, layer1, layer2, output_layer);
		if(prediction == ind) {
			accuracy++;
		}
		te++; 
	}
	accuracy /= validation_size;
	
	std::cout << "\nValidation Accuracy: " << accuracy << "\n";
	//}
	// nan test block
	/*
	read_mnist_train_image(tif, image, 5000);
	ind = read_mnist_train_label(tlf, 5000);
	label[ind] = 1.0;
	std::cout << "\nLabel: ";
	for(int i = 0; i < 28; i++) {
		for(int j = 0; j < 28; j++) {
			if(image[i * 28 + j] > 0) std::cout << "#" << " ";
			else std::cout << "  ";
		}
		std::cout << std::endl;
	}	
	for(int i = 0; i < 100; i++) {
		forward(image, layer1, layer2, output_layer);
		backpropagation(image, label, layer1, layer2, output_layer);
	}
	double pred = output_layer.get_values()[0];
	int max_ind;
	for(int i = 0; i < output_layer.get_num_neurons(); i++) {
		if(pred < output_layer.get_values()[i]) {
			pred = output_layer.get_values()[i];
			max_ind = i;
		}
	}
	std::cout << "\nPrediction: " << max_ind << "\n";
	*/
	
	tif.close();
	tlf.close();
		
	
	return 0;
}
