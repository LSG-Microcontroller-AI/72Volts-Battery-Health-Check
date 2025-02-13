#include <iostream>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

// Funzione ReLU
double relu(double x) {
	return (x > 0) ? x : 0;
}

// Derivata della funzione ReLU
double relu_derivative(double x) {
	return (x > 0) ? 1.0 : 0.0;
}


class NeuralNetwork {
public:
	// Dimensioni
	int input_size = 2;    // Due ingressi
	int hidden_size = 5;   // Cinque neuroni nascosti
	int output_size = 1;   // Un'uscita
	// Pesi e bias
	vector<vector<double>> weights_input_hidden;   // [input_size][hidden_size]
	vector<vector<double>> weights_hidden_output;    // [hidden_size][output_size]
	vector<double> bias_hidden;                      // [hidden_size]
	vector<double> bias_output;                      // [output_size]
	// Variabili per memorizzare i risultati del forward pass
	vector<vector<double>> hidden_layer_input;   // [n_samples][hidden_size]
	vector<vector<double>> hidden_layer_output;  // [n_samples][hidden_size]
	// Generatore di numeri casuali
	mt19937 gen;
	NeuralNetwork() {
		// Inizializza il generatore
		random_device rd;
		gen = mt19937(rd());
		normal_distribution<double> dist(0.0, 1.0);
		// Inizializza i vettori dei pesi e dei bias con le dimensioni corrette
		weights_input_hidden.resize(input_size, vector<double>(hidden_size));
		weights_hidden_output.resize(hidden_size, vector<double>(output_size));
		bias_hidden.resize(hidden_size, 0.0);
		bias_output.resize(output_size, 0.0);

		// Inizializzazione He per ReLU
		double init_scale_input = sqrt(2.0 / input_size);

		double init_scale_hidden = sqrt(2.0 / hidden_size);

		// Inizializza weights_input_hidden
		for (int i = 0; i < input_size; i++) {
			for (int j = 0; j < hidden_size; j++) {
				weights_input_hidden[i][j] = dist(gen) * init_scale_input;
			}
		}

		// Inizializza weights_hidden_output
		for (int i = 0; i < hidden_size; i++) {
			for (int j = 0; j < output_size; j++) {
				weights_hidden_output[i][j] = dist(gen) * init_scale_hidden;
			}
		}
	}

	// Funzione forward: calcola l'output della rete dato l'input
	// input_data: vettore di vettori con dimensioni [n_samples][input_size]
	// Restituisce un vettore di vettori [n_samples][output_size]
	vector<vector<double>> forward(const vector<vector<double>>& input_data) {
		int n_samples = input_data.size();
		// Calcola il layer nascosto
		hidden_layer_input.assign(n_samples, vector<double>(hidden_size, 0.0));
		hidden_layer_output.assign(n_samples, vector<double>(hidden_size, 0.0));
		for (int sample = 0; sample < n_samples; sample++) {
			for (int j = 0; j < hidden_size; j++) {
				double sum = 0.0;
				for (int i = 0; i < input_size; i++) {
					sum += input_data[sample][i] * weights_input_hidden[i][j];
				}
				// Aggiungo il bias per il neurone j
				hidden_layer_input[sample][j] = sum + bias_hidden[j];
				// Applica la funzione di attivazione ReLU
				hidden_layer_output[sample][j] = relu(hidden_layer_input[sample][j]);
			}
		}
		// Calcola il layer di output (attivazione lineare)
		vector<vector<double>> network_data_output(n_samples, vector<double>(output_size, 0.0));
		for (int sample = 0; sample < n_samples; sample++) {
			for (int j = 0; j < output_size; j++) {
				double sum = 0.0;
				for (int h = 0; h < hidden_size; h++) {
					sum += hidden_layer_output[sample][h] * weights_hidden_output[h][j];
				}
				network_data_output[sample][j] = sum + bias_output[j];
			}
		}
		return network_data_output;
	}
	// Funzione di backpropagation
	void backward(const vector<vector<double>>& input_training_data,
		const vector<vector<double>>& output_training_data,
		const vector<vector<double>>& network_data_output,
		double lr = 0.01) {
		int n_samples = input_training_data.size();
		// Calcola l'errore della rete: (target - output)
		vector<vector<double>> network_error(n_samples, vector<double>(output_size, 0.0));
		for (int sample = 0; sample < n_samples; sample++) {
			for (int j = 0; j < output_size; j++) {
				network_error[sample][j] = output_training_data[sample][j] - network_data_output[sample][j];
			}
		}
		// Per l'output lineare la derivata è 1, quindi:
		vector<vector<double>> delta_output = network_error; // [n_samples][output_size]
		// Calcola il gradiente per il layer nascosto
		vector<vector<double>> error_hidden_layer(n_samples, vector<double>(hidden_size, 0.0));
		vector<vector<double>> d_hidden_layer(n_samples, vector<double>(hidden_size, 0.0));
		for (int sample = 0; sample < n_samples; sample++) {
			for (int h = 0; h < hidden_size; h++) {
				// Poiché weights_hidden_output[h][j] e output_size è 1, possiamo semplificare:
				error_hidden_layer[sample][h] = delta_output[sample][0] * weights_hidden_output[h][0];
				// Moltiplica per la derivata di ReLU
				d_hidden_layer[sample][h] = error_hidden_layer[sample][h] * relu_derivative(hidden_layer_input[sample][h]);
			}
		}
		// Aggiornamento dei pesi per weights_hidden_output:
		// weights_hidden_output[h][j] += lr * sum_over_samples( hidden_layer_output[sample][h] * delta_output[sample][j] )
		for (int h = 0; h < hidden_size; h++) {
			double gradient = 0.0;
			for (int sample = 0; sample < n_samples; sample++) {
				gradient += hidden_layer_output[sample][h] * delta_output[sample][0];
			}
			weights_hidden_output[h][0] += lr * gradient;
		}
		// Aggiornamento dei pesi per weights_input_hidden:
		// weights_input_hidden[i][h] += lr * sum_over_samples( input_training_data[sample][i] * d_hidden_layer[sample][h] )
		for (int i = 0; i < input_size; i++) {
			for (int h = 0; h < hidden_size; h++) {
				double gradient = 0.0;
				for (int sample = 0; sample < n_samples; sample++) {
					gradient += input_training_data[sample][i] * d_hidden_layer[sample][h];
				}
				weights_input_hidden[i][h] += lr * gradient;
			}
		}
		// Aggiornamento del bias per il layer di output:
		// bias_output[j] += lr * sum_over_samples( delta_output[sample][j] )
		{
			double gradient = 0.0;
			for (int sample = 0; sample < n_samples; sample++) {
				gradient += delta_output[sample][0];
			}
			bias_output[0] += lr * gradient;
		}
		// Aggiornamento del bias per il layer nascosto:
		// bias_hidden[h] += lr * sum_over_samples( d_hidden_layer[sample][h] )
		for (int h = 0; h < hidden_size; h++) {
			double gradient = 0.0;
			for (int sample = 0; sample < n_samples; sample++) {
				gradient += d_hidden_layer[sample][h];
			}
			bias_hidden[h] += lr * gradient;
		}
	}
	// Funzione di training
	void training(const vector<vector<double>>& input_training_data,
		const vector<vector<double>>& output_training_data,
		int epochs, double lr = 0.01) {
		int n_samples = input_training_data.size();
		for (int epoch = 0; epoch < epochs; epoch++) {
			// Forward pass
			vector<vector<double>> output = forward(input_training_data);
			// Backward pass
			backward(input_training_data, output_training_data, output, lr);
			// Ogni 1000 epoche calcola e mostra la loss (errore quadratico medio)
			if (epoch % 1000 == 0) {
				double loss = 0.0;
				for (int sample = 0; sample < n_samples; sample++) {
					double err = output[sample][0] - output_training_data[sample][0];
					loss += err * err;
				}
				loss /= n_samples;
				cout << "Epoch " << epoch << " - Loss: " << loss << endl;
			}
		}
	}
};
int main() {
	// Prepara i dati di training:
	// Genera coppie (i, j) per i, j in [0, 5] e normalizza dividendo per 10.0
	int samples_per_side = 6;
	int total_samples = samples_per_side * samples_per_side;
	vector<vector<double>> input_training_data(total_samples, vector<double>(2, 0.0));
	vector<vector<double>> output_training_data(total_samples, vector<double>(1, 0.0));
	int index = 0;
	for (int i = 0; i < samples_per_side; i++) {
		for (int j = 0; j < samples_per_side; j++) {
			input_training_data[index][0] = i / 10.0;
			input_training_data[index][1] = j / 10.0;
			output_training_data[index][0] = (i + j) / 10.0;
			index++;
		}
	}
	// Inizializza e allena la rete neurale
	NeuralNetwork nn;
	nn.training(input_training_data, output_training_data, 10000, 0.01);
	// Test con un nuovo input (normalizzato)
	vector<vector<double>> test_input = { {2 / 10.0, 3 / 10.0} };
	vector<vector<double>> test_output = nn.forward(test_input);
	// Riporta alla scala originale
	double output_visualization = test_output[0][0] * 10;
	cout << "Input: [ " << test_input[0][0] * 10 << ", " << test_input[0][1] * 10 << " ]"
		<< " --> Output: " << output_visualization << endl;
	return 0;
}
