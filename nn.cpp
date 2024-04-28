// Neural network implementation

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

#define alpha 0.001
#define epochs 100
#define training_size 60000

const int IMAGE_SIZE = 28 * 28; // Each image is 28x28 pixels
const int LABEL_SIZE = 1;

double sigmoid(double x) {
	return (double) 1 / (1 + exp(-x));
}

double relu(double x) {
	return x > 0 ? x : 0;
}

bool readMNISTFile(const std::string& filename, std::vector<double>& image, int& label) {
    // Open the MNIST file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    try {
        // Read image data (pixels)
        std::vector<unsigned char> image_bytes(IMAGE_SIZE);
        file.read(reinterpret_cast<char*>(image_bytes.data()), IMAGE_SIZE);

        // Convert image data to double values (normalize to range [0, 1])
        image.resize(IMAGE_SIZE);
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            image[i] = static_cast<double>(image_bytes[i]) / 255.0; // Normalize to [0, 1]
        }

        // Read label data
        file.read(reinterpret_cast<int*>(&label), LABEL_SIZE);

        // Close the file
        file.close();
    } catch (const std::exception& e) {
        std::cerr << "Error reading file: " << filename << std::endl;
        return false;
    }

    return true;
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
		for(int i = 0; i < K; i++) {
			weights.push_back(dist(gen));
			// bias.push_back(dist(gen));
		}
		value = 0;
	}
	
	void calculate_value(std::vector<double>& x, int K) {
		for(int i = 0; i < K; i++) {
			value += weight[i] * x[i] + bias[i];
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
	
	~neuron();
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
		for(neuron n:neurons) {
			n.calculate_value(x, K)
			values.append(n.get_value());
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
		for(neuron n:neurons) {
			y = n.get_value();
			loss += (t - y) * (t - y);
		}
		loss /= 2;
	}
	
	void update_weights(std::vector<double>& x, std::vector<double>& t, int K) {
		for(neuron n:neurons) {
			n.update_weights(x, t, K);
		}
	}
	
	~layer();
};

/*class sequential_nn {
	std::vector<layer> layers;
	
	public:
	sequential_nn(int input, int n1, int n2, int n3) {
		layers.append(layer(input, n1));
		layers.append(layer(n1, n2));
		layers.append(layer(n2, n3));
	}
};*/
	

int main() {
	std::vector<double> image;
	
	std::vector<double> labels;
	labels.resize(training_size);
	
	// std::vector<double> predictions;
	// predictions.resize(training_size);
	
	// Initializing the layers
	layer layer1 = layer(image.size(), 128);
	layer layer2 = layer(128, 64);
	layer layer3 = layer(64, 10);
	layer output_layer = layer(10, 1);
	
	std::ofstream loss("./loss.txt");
	
	int ep = 0;
	
	while(ep < epochs) {
		// Assuming image has stored 1 image data from the MNIST dataset
		int i = 0;
		while(i < training_size) {
			readMNISTFile("", image, labels[i]);
			
			layer1.calculate_values(image, image.size());
			layer2.calculate_values(layer1.get_values(), layer1.get_values().size());
			layer3.calculate_values(layer2.get_values(), layer2.get_values().size());
			output_layer.calculate_values(layer3.get_values(), layer3.get_values().size());
			
			// predictions[i] = output_layer.get_values()[0];
			std::vector<double> temp_label = {labels[i]};
			output_layer.update_weights(layer3.get_values(), temp_label, layer3.get_values().size());
			layer3.update_weights(layer2.get_values(), output_layer.get_values(), layer2.get_values().size());
			layer2.update_weights(layer1.get_values(), layer3.get_values(), layer1.get_values().size());
			layer1.update_weights(image, layer2.get_values(), image.size());
			
			i++;
		}
		
		loss << output_layer.get_values()[0] << std::endl;
		
		ep++;
	}
	
	file.close();
		
	
	return 0;
}
