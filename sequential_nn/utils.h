#ifndef UTILS
#define UTILS

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

#endif
