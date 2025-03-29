using namespace std;
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <stdint.h>
#include <sstream>
#include <chrono>
#include <ctime>
#include <string>
#include <fstream>
#include <cfloat>
#include <random>
#include <vector>
#include <conio.h>
#include <windows.h>
void init();
void lavora();
double get_random_number_from_xavier();
void forward();
void apprendi();
void back_propagate();
void read_weights_from_file();
void write_weights_on_file();
void read_samples_from_file_diagram_battery();
float sigmoid_activation(float A);
float _err_epoca;
float _err_rete = 0.00f;
float _err_amm = 0.00025f;
float _epsilon = 0.10f;
uint16_t const training_samples = 101;
const uint8_t numberOf_X = 2;
const uint8_t numberOf_H = 10;
const uint8_t numberOf_Y = 6;
float output_bias[numberOf_Y] = { 0.00 };
float hidden_bias[numberOf_H] = { 0.00 };
double _lower_bound_xavier;
double _upper_bound_xavier;
float W1[numberOf_X][numberOf_H] = { 0.00 };
float W2[numberOf_H][numberOf_Y] = { 0.00 };
float x[numberOf_X] = { 0.00 };
float h[numberOf_H] = { 0.00 };
float y[numberOf_Y] = { 0.00 };
float d[numberOf_Y] = { 0.00 };
float amps_training[training_samples]{};
float watts_hour_training[training_samples]{};
float battery_out_training[training_samples][numberOf_Y]{};
string global_time_recorded;
default_random_engine generator(time(0));
const string _relative_files_path = "72V-Battery-S11";
int _epoca_index = 0;
float err_min_rete = FLT_MAX;
bool is_on_wtrite_file = false;
int main() {
	init();
	char response;
	cout << "\n Do you want load the weights file\n";
#ifdef __linux__
	response = std::cin.get();
	std::cin.ignore();
#elif _WIN32
	response = _getch();
#else
#endif
	if (response == 'y') {
		cout << "\n Weights loaded\n";

		read_weights_from_file();
	}
	else {
		cout << "\n Weights overwritten\n";
	}
	cout << "\n Do you want to start learning\n";
#ifdef __linux__
	response = std::cin.get();
	std::cin.ignore();
#elif _WIN32
	response = _getch();
#else
#endif
	if (response == 'y') {
		cout << "\n Start to learning......\n";
		std::cout << "Inserisci il valore per epsilon: ";
		std::cin >> _epsilon;
		apprendi();
	}
	lavora();
	cout << "press a key..\n\n";
}
double xavier_init(double n_x, double n_y) {
	return sqrt(6.0) / sqrt(n_x + n_y);
}
void init() {
	double param = xavier_init(numberOf_X, numberOf_Y);
	cout << "xavier glorot param : " << param << "\n\n";
	_lower_bound_xavier = -param;
	_upper_bound_xavier = param;
	//-----------------------------------	bias initialization
	for (int i = 0; i < numberOf_Y; i++) {
		output_bias[i] = 0.1f;
	}
	for (int i = 0; i < numberOf_H; i++) {
		hidden_bias[i] = 0.1f;
	}
	//-----------------------------------	console input values + Hidden bias values
	//cout << "input elements initialization:\n\n";
	for (int i = 0; i < (numberOf_X); i++) {
		//x[i] = 0.00f;
		cout << "x[" << i << "]" << "=" << x[i] << "\n";
	}
	for (int i = 0; i < numberOf_H; i++) {
		cout << "hidden_bias[" << i << "]" << "=" << hidden_bias[i] << "-BIAS" << "\n";
	}
	//-----------------------------------	console hidden values + output bias values
	for (int i = 0; i < (numberOf_H); i++) {
		//h[i] = 0.00f;
		cout << "h[" << i << "]" << "=" << h[i] << "\n";
	}
	for (int i = 0; i < numberOf_Y; i++) {
		cout << "output_bias[" << i << "]" << "=" << output_bias[i] << "-BIAS" << "\n";
	}
	//cout << "output elements initialization:\n\n";
	//-----------------------------------	console output values
	for (int i = 0; i < numberOf_Y; i++) {
		//y[i] = 0.00f;
		cout << "y[" << i << "]=" << y[i] << "\n";
	}
	//-----------------------------------	console W1 values
	cout << "W1 elements initialization:\n\n";
	for (int i = 0; i < numberOf_X; i++) {
		for (int k = 0; k < numberOf_H; k++) {
			W1[i][k] = get_random_number_from_xavier();
			cout << "W1[" << i << "]" << "[" << k << "]" << "=" << W1[i][k] << "\n";
		}
	}
	//-----------------------------------	console W2 values
	for (int k = 0; k < numberOf_H; k++) {
		for (int j = 0; j < numberOf_Y; j++) {
			W2[k][j] = get_random_number_from_xavier();
			cout << "W2[" << k << "]" << "[" << j << "]" << "=" << W2[k][j] << "\n";
		}
	}
}
void lavora() {
	while (true) {
		// Messaggio iniziale
		std::cout << "\nInserisci i valori per Ampere e Watt-ora (Ctrl+C per uscire):\n";
		// Input per x[0] (Ampere)
		std::cout << "Inserisci il valore per x[0] (Ampere, float): ";
		std::cin >> x[0];
		// Input per x[1] (Watt-ora)
		std::cout << "Inserisci il valore per x[1] (Watt-ora, float): ";
		std::cin >> x[1];
		// Stampa dei valori
		std::cout << "\nHai inserito:\n";
		std::cout << "x[0] (Ampere) = " << x[0] << "\n";
		std::cout << "x[1] (Watt-ora) = " << x[1] << "\n";
		x[0] = log(x[0] + 1.0f) / 10.0f;
		x[1] = log(x[1] + 1.0f) / 10.0f;
		forward();
		// Stampa dei risultati
		std::cout << "\n x[0] = " << exp(x[0] * 10) << " x[1] = " << exp(x[1] * 10) << "\n"
			<< "\n y[0] = " << y[0] * 10.00f
			<< "\n y[1] = " << y[1] * 10.00f
			<< "\n y[2] = " << y[2] * 10.00f
			<< "\n y[3] = " << y[3] * 10.00f
			<< "\n y[4] = " << y[4] * 10.00f
			<< "\n y[5] = " << y[5] * 10.00f;
	}
}
void apprendi() {
	int cout_counter = 0;
	auto start = std::chrono::system_clock::now();
	read_samples_from_file_diagram_battery();
	float err_epoca_min_value = FLT_MAX;
	do {
		_err_epoca = 0.00f;
		for (unsigned long p = 0; p < training_samples; p++) {
			x[0] = log(amps_training[p] + 1.0f) / 10.0f;
			x[1] = log(watts_hour_training[p] + 1.0f) / 10.0f;
			for (int i = 0; i < numberOf_Y; i++) {
				d[i] = battery_out_training[p][i] / 10.00f;
			}
			forward();
			back_propagate();
			if (_err_rete > _err_epoca) {
				_err_epoca = _err_rete;
			}
			_epoca_index++;
		}
		cout_counter++;
		is_on_wtrite_file = false;
		if (cout_counter == 100000) {
			std::cout << "\nepoca:" << _epoca_index <<
				"\nerr_epoca=" << _err_epoca << "\n"
				"epsilon=" << _epsilon << "\n";
			cout_counter = 0;
			if (err_epoca_min_value > _err_epoca) {
				is_on_wtrite_file = true;
				err_epoca_min_value = _err_epoca;
			}
		}
		if (is_on_wtrite_file) {
			std::time_t now = std::time(nullptr);
			std::tm local_time;
			err_min_rete = _err_rete;
			std::cout << "\nwrite on file\n";
			write_weights_on_file();
		}
	} while (_err_epoca > _err_amm);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	double sample_time = elapsed_seconds.count();
	std::cout << "learning time : " << (int)(sample_time / 60) << " minutes.\n";
#ifdef __linux__
	getchar();
#elif _WIN32
	_getch();
#endif
}
void forward() {
	for (int k = 0; k < (numberOf_H); k++) {
		float Zk = 0.00f;
		for (int i = 0; i < numberOf_X; i++) {
			Zk += (W1[i][k] * x[i]);
		}
		//insert X bias
		Zk += hidden_bias[k];
		h[k] = sigmoid_activation(Zk);
	}
	for (int j = 0; j < numberOf_Y; j++) {
		float Zj = 0.00f;
		for (int k = 0; k < numberOf_H; k++) {
			Zj += (W2[k][j] * h[k]);
		}
		//insert H bias
		Zj += output_bias[j];
		y[j] = sigmoid_activation(Zj);
	}
}
void back_propagate() {
	float err_H[numberOf_H] = { 0.00f };
	float delta = 0.00f;
	_err_rete = 0.00f;
	for (int j = 0; j < numberOf_Y; j++) {
		if (abs(d[j] - y[j]) > _err_rete) {
			_err_rete = abs(d[j] - y[j]);
		}
		delta = (d[j] - y[j]) * y[j] * (1.00f - y[j]);
		for (int k = 0; k < numberOf_H; k++) {
			W2[k][j] += (_epsilon * delta * h[k]);
			err_H[k] += (delta * W2[k][j]);
		}
		output_bias[j] += _epsilon * delta;
	}
	for (int k = 0; k < numberOf_H; k++) {
		delta = err_H[k] * h[k] * (1.00f - h[k]);
		for (int i = 0; i < numberOf_X; i++) {
			W1[i][k] += (_epsilon * delta * x[i]);
		}
		hidden_bias[k] += _epsilon * delta;
	}
}
double get_random_number_from_xavier() {
	uniform_real_distribution<double> distribution(_lower_bound_xavier, _upper_bound_xavier);
	double random_value = distribution(generator);
	return random_value;
}
float sigmoid_activation(float Z) {
	return 1.00f / (1.00f + exp((Z * -1)));
}
void read_samples_from_file_diagram_battery() {
	std::string filename = _relative_files_path + "/" + "72V_Battery.CSV";//"72V_Battery.CSV";
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Errore nell'apertura del file " << filename << std::endl;
	}
	std::string line;
	int training_block_index = 0;
	int training_row_index = 0;
	int training_row_pre_index = 0;
	std::string item;
	stringstream ss1;
	while (!file.eof()) {
		training_row_pre_index = training_row_index++;
		switch (training_row_pre_index) {
		case 0:
		case 1:
		case 2:
		case 3:
		case 4:
		case 5:
			try {
				std::getline(file, line);
				ss1.str(line);
				std::getline(ss1, item, ';');
				std::getline(ss1, item, ';');
				std::getline(ss1, item, ';');
				battery_out_training[training_block_index][training_row_pre_index] = std::stod(item);
				cout << "battery[" << training_block_index << "]" << "[" << training_row_pre_index << "] = " << battery_out_training[training_block_index][training_row_pre_index] << "\n";
			}
			catch (...) {
				cout << "--------------------------errore in lettura file";
			};
			break;
		case 6:
			try {
				std::getline(file, line);
				ss1.str(line);
				std::getline(ss1, item, ';');
				std::getline(ss1, item, ';');
				std::getline(ss1, item, ';');
				watts_hour_training[training_block_index] = std::stod(item);
				cout << "Watts/hour[" << training_block_index << "] = " << watts_hour_training[training_block_index] << "\n";
			}
			catch (...) {
				cout << "----------------------------------errore in lettura file";
			};
			break;
		case 7:
			try {
				std::getline(file, line);
				ss1.str(line);
				std::getline(ss1, item, ';');
				std::getline(ss1, item, ';');
				std::getline(ss1, item, ';');
				amps_training[training_block_index] = std::stod(item);
				cout << "Ampere[" << training_block_index << "] = " << amps_training[training_block_index] << "\n";
			}
			catch (...) {
				cout << "-------------------------------------------errore in lettura file";
			};
			break;
		default:
			training_block_index++;
			training_row_index = 0;
			break;
		}
	}
#ifdef __linux__
#elif _WIN32
	//system("pause");
#else
#endif
	if (training_block_index != training_samples) {
		cout << "\n\nALLERT!!!!!!! training sample different to index = \t" << training_block_index << "\n";
#ifdef __linux__
#elif _WIN32
		system("pause");
#else
#endif
	}
	else {
		cout << "\n\nTraining sample index is " << training_block_index << " and seems to have been loaded correctly.";
#ifdef __linux__
#elif _WIN32
		system("pause");
#else

#endif
	}
	file.close();
}
void read_weights_from_file() {
	std::ifstream in(_relative_files_path + "/" + "model.bin", std::ios_base::binary);
	if (in.good()) {
		for (int i = 0; i < numberOf_X; i++) {
			for (int k = 0; k < numberOf_H; k++) {
				in.read((char*)&W1[i][k], sizeof(float));
			}
		}
		for (int j = 0; j < numberOf_Y; j++) {
			for (int k = 0; k < numberOf_H; k++) {
				in.read((char*)&W2[k][j], sizeof(float));
			}
		}
		for (int k = 0; k < numberOf_H; k++) {
			in.read((char*)&hidden_bias[k], sizeof(float));
		}
		for (int j = 0; j < numberOf_Y; j++) {
			in.read((char*)&output_bias[j], sizeof(float));
		}
	}
}
void write_weights_on_file() {
	std::ofstream fw(_relative_files_path + "/" + "model.bin", std::ios_base::binary);
	if (fw.good()) {
		for (int i = 0; i < numberOf_X; i++) {
			for (int k = 0; k < numberOf_H; k++) {
				fw.write((char*)&W1[i][k], sizeof(float));
			}
		}
		for (int j = 0; j < numberOf_Y; j++) {
			for (int k = 0; k < numberOf_H; k++) {
				fw.write((char*)&W2[k][j], sizeof(float));
			}
		}
		for (int k = 0; k < numberOf_H; k++) {
			fw.write((char*)&hidden_bias[k], sizeof(float));
		}
		for (int j = 0; j < numberOf_Y; j++) {
			fw.write((char*)&output_bias[j], sizeof(float));
		}
		fw.write((char*)&_err_epoca, sizeof(float));
		fw.close();
	}
	else {
		cout << "Problem with opening file";
	}
}





