// Neural network implementation

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

#include "sequential_nn/sequential_nn.h"

const int epochs = 5;
const int total_size = 600;
const int training_size = 500;
const int validation_size = total_size - training_size;
const int image_size = 28 * 28;
bool verbose = 0;

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

int main() {
	std::vector<double> image(image_size);
	std::vector<double> label(10, 0.0);
	int ind;
	
	std::string train_image_file = "data/train-images-idx3-ubyte";
	std::string train_label_file = "data/train-labels-idx1-ubyte";
	std::ifstream tif(train_image_file, std::ios::binary);
	std::ifstream tlf(train_label_file, std::ios::binary);
	
	int layer1_neurons = 128;
	int layer2_neurons = 64;
	int output_layer_neurons = 10;
	
	// Initializing the layers
	std::vector<layer> layers;
	layers.emplace_back(image_size, layer1_neurons, "relu");
	layers.emplace_back(layer1_neurons, layer2_neurons, "relu");
	layers.emplace_back(layer2_neurons, output_layer_neurons, "softmax");
	
	sequential_nn nn = sequential_nn(layers);
	
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
			
			nn.fit(image, label);
			
			label[ind] = 0.0;
			avg_loss += nn.get_loss();
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
		prediction = nn.predict(image);
		if(prediction == ind) {
			accuracy++;
		}
		te++; 
	}
	accuracy /= validation_size;
	
	std::cout << "\nValidation Accuracy: " << accuracy << "\n";
	
	tif.close();
	tlf.close();
		
	
	return 0;
}
