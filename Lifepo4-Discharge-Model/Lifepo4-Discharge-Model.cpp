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
//#include "imgui.h"
//#include "implot.h"
//#include "imgui_impl_glfw.h"
//#include "imgui_impl_opengl3.h"
//#include <GLFW/glfw3.h>
#include <vector>
#include "plot_renderer.h"
#ifdef __linux__
#elif _WIN32
#include <conio.h>
#include <windows.h>
#else
#endif
void init();
void predict();
void forward();
void apprendi();
void back_propagate();
void read_weights_from_file();
void write_weights_on_file();
void read_samples_from_file_diagram_battery();
float _err_epoca;
float _max_single_traning_output_error = 0.00f;
float _err_amm = 0.009f;
float _epsilon = 0.001f;
uint16_t const training_samples = 338;
const uint8_t numberOf_X = 2;
const uint8_t numberOf_H = 25;
const uint8_t numberOf_Y = 6;
float output_bias[numberOf_Y] = { 0.00 };
float hidden_bias[numberOf_H] = { 0.00 };
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
//GLFWwindow* window;
//std::vector<double> ascissa1;
//std::vector<double> ascissa2;
//std::vector<double> ascissa3;
//std::vector<double> ascissa4;
//std::vector<double> ordinata1;
//std::vector<double> ordinata2;
//std::vector<double> ordinata3;
//std::vector<double> ordinata4;
int _epoca_index = 0;
mt19937 gen;
float err_min_rete = FLT_MAX;
bool is_on_wtrite_file = false;
float _max_single_traning_output_error_average = 0.00f;
float _err_epoca_min_value = FLT_MAX;
float relu(float x) {
	return (x > 0) ? x : 0;
}
// Derivata della funzione ReLU
//GLFWwindow* InitWindow() {
//	if (!glfwInit()) {
//		return nullptr;
//	}
//	// Abilita l'hint per una finestra massimizzata
//	glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
//	// Crea la finestra con dimensioni standard
//	GLFWwindow* window = glfwCreateWindow(1280, 720, "Grafici Seno e Coseno", NULL, NULL);
//	if (!window) {
//		glfwTerminate();
//		return nullptr;
//	}
//	// Ora massimizziamo la finestra dopo la creazione
//	glfwMaximizeWindow(window);
//	glfwMakeContextCurrent(window);
//	glfwSwapInterval(1);
//	// Inizializzazione ImGui + ImPlot
//	IMGUI_CHECKVERSION();
//	ImGui::CreateContext();
//	ImPlot::CreateContext();
//	ImGui_ImplGlfw_InitForOpenGL(window, true);
//	ImGui_ImplOpenGL3_Init("#version 130");
//	return window;
//}
//void open_plots(PlotRenderer plot1, PlotRenderer plot2, PlotRenderer plot3, PlotRenderer plot4) {
//	glfwPollEvents();
//	ImGui_ImplOpenGL3_NewFrame();
//	ImGui_ImplGlfw_NewFrame();
//	ImGui::NewFrame();
//	plot1.Begin();
//	plot2.Begin();
//	plot3.Begin();
//	plot4.Begin();
//	ImGui::Render();
//	int display_w, display_h;
//	glfwGetFramebufferSize(window, &display_w, &display_h);
//	glViewport(0, 0, display_w, display_h);
//	glClear(GL_COLOR_BUFFER_BIT);
//	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
//	glfwSwapBuffers(window);
//	GLenum err;
//	while ((err = glGetError()) != GL_NO_ERROR) {
//		std::cout << "Errore OpenGL: " << err << std::endl;
//	}
//}
int main(){
	//window = InitWindow();
#ifdef __linux__
#elif _WIN32
	HWND consoleWindow = GetConsoleWindow();
	// Sposta e massimizza la finestra
	SetWindowPos(consoleWindow, nullptr, -1920, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
	ShowWindow(consoleWindow, SW_MAXIMIZE);
#endif
	init();
#ifdef __linux__
	// no sound on linux
#elif _WIN32
	Beep(3000, 200);
#endif
	char response;
	cout << "\n'n' for new learning, 'c' for continue learning, 'e' for execute.\n";
	//cout << "\n Do you want load the weights file\n";
#ifdef __linux__
	response = std::cin.get();
	std::cin.ignore();
#elif _WIN32
	response = _getch();
#endif
	if (response == 'e'){
		read_weights_from_file();
		predict();
	}
	if (response == 'c'){
		cout << "\nmodel loaded.\n";
		read_weights_from_file();
		cout << "\nlast inserted epsilon is : " << _epsilon <<
			" do you wanna change it? y or n\n"; 
#ifdef __linux__
		response = std::cin.get();
		std::cin.ignore();
#elif _WIN32
		response = _getch();
#endif
		if (response == 'y'){
			cout << "\ninsert new epsilon value : ";
			cin >> _epsilon;
			cout << "\n epsilon changed\n";
		}
		else{
			cout << "\n epsilon not changed\n";
		}
		apprendi();
	}
	if (response == 'n'){
		cout << "ATTENTION !!!!!!!!!!!!! are you sure to restart learning? press y to continue !!!!!!!!!!!!!!!!!!!\n\n";
#ifdef __linux__
		response = std::cin.get();
		std::cin.ignore();
#elif _WIN32
		response = _getch();
#endif
		if (response == 'y'){
			cout << "\n model overwritten\n";
			apprendi();
		}
		else{
			cout << "\nprocess blocked.\n";
			/*read_weights_from_file();
			apprendi();*/
		}
	}
}
void init()
{
	random_device rd;
	gen = mt19937(rd());
	double init_scale_input = sqrt(2.0 / numberOf_Y);
	double init_scale_hidden = sqrt(2.0 / numberOf_H);
	normal_distribution<double> dist(0.0, 1.0);
	//-----------------------------------	bias initialization
	for (int i = 0; i < numberOf_Y; i++)
	{
		output_bias[i] = 0.1f;
	}
	for (int i = 0; i < numberOf_H; i++)
	{
		hidden_bias[i] = 0.1f;
	}
	//-----------------------------------	console input values + Hidden bias values
	//cout << "input elements initialization:\n\n";
	for (int i = 0; i < (numberOf_X); i++)
	{
		//x[i] = 0.00f;
		cout << "x[" << i << "]" << "=" << x[i] << "\n";
	}
	for (int i = 0; i < numberOf_H; i++)
	{
		cout << "hidden_bias[" << i << "]" << "=" << hidden_bias[i] << "-BIAS" << "\n";
	}
	//-----------------------------------	console hidden values + output bias values
	for (int i = 0; i < (numberOf_H); i++)
	{
		//h[i] = 0.00f;
		cout << "h[" << i << "]" << "=" << h[i] << "\n";
	}
	for (int i = 0; i < numberOf_Y; i++)
	{
		cout << "output_bias[" << i << "]" << "=" << output_bias[i] << "-BIAS" << "\n";
	}
	//cout << "output elements initialization:\n\n";
	//-----------------------------------	console output values
	for (int i = 0; i < numberOf_Y; i++)
	{
		//y[i] = 0.00f;
		cout << "y[" << i << "]=" << y[i] << "\n";
	}
	//-----------------------------------	console W1 values
	cout << "W1 elements initialization:\n\n";
	for (int i = 0; i < numberOf_X; i++)
	{
		for (int k = 0; k < numberOf_H; k++)
		{
			W1[i][k] = dist(gen) * init_scale_input;
			cout << "W1[" << i << "]" << "[" << k << "]" << "=" << W1[i][k] << "\n";
		}
	}
	//-----------------------------------	console W2 values
	for (int k = 0; k < numberOf_H; k++)
	{
		for (int j = 0; j < numberOf_Y; j++)
		{
			W2[k][j] = dist(gen) * init_scale_hidden;
			cout << "W2[" << k << "]" << "[" << j << "]" << "=" << W2[k][j] << "\n";
		}
	}
}
void predict()
{
	while (true)
	{
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
		x[0] = log(x[0] + 1.00f) / 10.00f;
		x[1] = log(x[1] + 1.00f) / 10.00f;
		forward();
		// Stampa dei risultati
		std::cout << "\n x[0] = " << exp(x[0] * 10.00f) << " x[1] = " << exp(x[1] * 10.00f) << "\n"
			<< "\n y[0] = " << y[0] * 10.00f
			<< "\n y[1] = " << y[1] * 10.00f
			<< "\n y[2] = " << y[2] * 10.00f
			<< "\n y[3] = " << y[3] * 10.00f
			<< "\n y[4] = " << y[4] * 10.00f
			<< "\n y[5] = " << y[5] * 10.00f;
	}
}
//void print_graph(const char* window_name, float ordinata, const char description_ordinata[20], float ascissa, const char description_ascissa[20]) {
//	ascissa1.push_back(ascissa);
//	ordinata1.push_back(ordinata);
//	/*char title_ordinata[30] = "epoca vs ";
//	size_t available = sizeof(title_ordinata) - strlen(title_ordinata) - 1;
//	errno_t err;
//	err = strncat_s(title_ordinata, sizeof(title_ordinata), description_ordinata, available);
//	if (err != 0) {
//		cout <<"Errore nella concatenazione: codice %d\n" << err;
//	}*/
//	PlotRenderer plot1(window_name, ascissa1, ordinata1, description_ascissa, description_ordinata, "Andamento Errore_rete");
//	open_plots(plot1, PlotRenderer(), PlotRenderer(), PlotRenderer());
//}
void apprendi() {
	int cout_counter = 0;
	auto start = std::chrono::system_clock::now();
	read_samples_from_file_diagram_battery();
	
	do {
		_err_epoca = 0.00f;
		_max_single_traning_output_error_average = 0.00f;
		uint16_t max_traning_sample_error = 0;
		for (unsigned long p = 0; p < training_samples; p++)
		{
			x[0] = log(amps_training[p] + 1.0f) / 10.0f;
			x[1] = log(watts_hour_training[p] + 1.0f) / 10.0f;
			for (int i = 0; i < numberOf_Y; i++) {
				d[i] = battery_out_training[p][i] / 10.00f;
			}
			forward();
			back_propagate();
			if (_max_single_traning_output_error > _err_epoca) {
				_err_epoca = _max_single_traning_output_error;
				max_traning_sample_error = training_samples;
			}
			_max_single_traning_output_error_average += _max_single_traning_output_error;
		}
		_epoca_index++;
		cout_counter++;
		is_on_wtrite_file = false;
		if (cout_counter == 1000) {
			_max_single_traning_output_error_average = _max_single_traning_output_error_average / training_samples;
			cout << "epoca : " << _epoca_index << "\n\n";
			cout << "media di max errore di traning : " << _max_single_traning_output_error_average << "\n\n";
			cout << "max errore di traning : " << _err_epoca << "\n\n";
			cout << "con epsilon : " << _epsilon << "\n\n";
			cout << "max error on traning sample : " << max_traning_sample_error << "\n\n";
			cout_counter = 0;
			if ((_err_epoca_min_value > _err_epoca) && _err_epoca > 0.00f){
				is_on_wtrite_file = true;
				_err_epoca_min_value = _err_epoca;
			}
		}
		if (is_on_wtrite_file)
		{
			std::time_t now = std::time(nullptr);
			std::tm local_time;
#ifdef __linux__
			if (localtime_r(&now, &local_time) == nullptr)
			{
				std::cerr << "Errore nella conversione del tempo.\n";
			}
			else
			{
				// Stampa il tempo locale in formato leggibile
				std::cout << "Anno: " << (1900 + local_time.tm_year) << "\n";
				std::cout << "Mese: " << (1 + local_time.tm_mon) << "\n";
				std::cout << "Giorno: " << local_time.tm_mday << "\n";
				std::cout << "Ora: " << local_time.tm_hour << "\n";
				std::cout << "Minuti: " << local_time.tm_min << "\n";
				std::cout << "Secondi: " << local_time.tm_sec << "\n";
			}
#elif _WIN32
			/*if (localtime_s(&local_time, &now) != 0) {
				std::cerr << "Errore nella conversione del tempo.\n";
			}*/
			/*global_time_recorded = std::to_string(local_time.tm_mday) + "/" +
				std::to_string(local_time.tm_mon + 1) + "/" +
				std::to_string(local_time.tm_year + 1900) + " " +
				std::to_string(local_time.tm_hour) + ":" +
				std::to_string(local_time.tm_min) + ":" +
				std::to_string(local_time.tm_sec);
			std::cout << "\nepoca:" << _epoca_index <<
				"\nlast modified date: " << global_time_recorded <<
				"\nerr_epoca=" << _err_epoca <<
				" min._err_epoca= " << err_epoca_min_value <<
				" _max_single_traning_output_error=" << _max_single_traning_output_error <<
				" min._max_single_traning_output_error= " << err_min_rete <<
				"\n";*/
#endif
			err_min_rete = _max_single_traning_output_error;
			std::cout << "\nwrite on file\n";
			write_weights_on_file();
		}
	} while (_err_epoca > _err_amm);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	double sample_time = elapsed_seconds.count();
	std::cout << "learning time : " << (int)(sample_time / 60) << " minutes.\n";
#ifdef __linux__
	// linux code goes here
#elif _WIN32
	Beep(3000, 200);
	Beep(3000, 200);
	Beep(3000, 200);
	Beep(3000, 200);
	Beep(3000, 200);
#endif
#ifdef __linux__
	getchar();
#elif _WIN32
	int response = _getch();
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
		h[k] = relu(Zk);
	}
	for (int j = 0; j < numberOf_Y; j++) {
		float Zj = 0.00f;
		for (int k = 0; k < numberOf_H; k++) {
			Zj += (W2[k][j] * h[k]);
		}
		//insert H bias
		Zj += output_bias[j];
		y[j] = Zj;
	}
}
void back_propagate() {
	float err_H[numberOf_H] = { 0.00f };
	float delta = 0.00f;
	_max_single_traning_output_error = 0.00f;
	// Calcolo del delta per il layer di output (attivazione lineare -> derivata = 1)
	float single_training_output_error = 0.00f;
	for (int j = 0; j < numberOf_Y; j++) {
		single_training_output_error = d[j] - y[j];
		if (fabs(single_training_output_error) > _max_single_traning_output_error) {
			_max_single_traning_output_error = fabs(single_training_output_error);
		}
		delta = single_training_output_error;  // Derivata del layer output lineare è 1
		// Aggiornamento dei pesi del layer di output e accumulo dell'errore per il layer nascosto
		for (int k = 0; k < numberOf_H; k++) {
			W2[k][j] += (_epsilon * delta * h[k]);
			err_H[k] += delta * W2[k][j];
		}
		// Aggiornamento del bias per il layer di output
		output_bias[j] += _epsilon * delta;
	}
	// Calcolo del delta per il layer nascosto usando la derivata della ReLU
	for (int k = 0; k < numberOf_H; k++) {
		// Derivata della ReLU: 1 se il neurone è attivo (h[k] > 0), 0 altrimenti
		float relu_deriv = (h[k] > 0.0f) ? 1.0f : 0.0f;
		delta = err_H[k] * relu_deriv;
		// Aggiornamento dei pesi del layer nascosto
		for (int i = 0; i < numberOf_X; i++) {
			W1[i][k] += (_epsilon * delta * x[i]);
		}
		// Aggiornamento del bias per il layer nascosto
		hidden_bias[k] += _epsilon * delta;
	}
}
void read_samples_from_file_diagram_battery()
{
	//std::cout << "Directory corrente: " << std::filesystem::current_path() << std::endl;
	std::string filename = _relative_files_path + "/" + "72V_Battery.CSV";//"72V_Battery.CSV";
	// Apertura del file
	std::ifstream file(filename);
	// Verifica se il file è stato aperto correttamente
	if (!file.is_open())
	{
		std::cerr << "Errore nell'apertura del file " << filename << std::endl;
	}
	std::string line;
	int training_block_index = 0;
	int training_row_index = 0;
	int training_row_pre_index = 0;
	std::string item;
	stringstream ss1;
	while (!file.eof())
	{
		training_row_pre_index = training_row_index++;
		switch (training_row_pre_index)
		{
		case 0:
		case 1:
		case 2:
		case 3:
		case 4:
		case 5:
			std::getline(file, line);
			ss1.str(line);
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			battery_out_training[training_block_index][training_row_pre_index] = std::stod(item);
			cout << "battery[" << training_block_index << "]" << "[" << training_row_pre_index << "] = " << battery_out_training[training_block_index][training_row_pre_index] << "\n";
			break;
		case 6:
			std::getline(file, line);
			ss1.str(line);
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			watts_hour_training[training_block_index] = std::stod(item);
			cout << "Watts/hour[" << training_block_index << "] = " << watts_hour_training[training_block_index] << "\n";
			break;
		case 7:

			std::getline(file, line);
			ss1.str(line);
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			amps_training[training_block_index] = std::stod(item);
			cout << "Ampere[" << training_block_index << "] = " << amps_training[training_block_index] << "\n";
			break;
		default:
			training_block_index++;
			training_row_index = 0;
			break;
		}
	}
#ifdef __linux__
#elif _WIN32
#else
#endif
	if ((training_block_index)+1 != training_samples)
	{
		cout << "\n\nALLERT!!!!!!! training sample different to index = \t" << training_block_index << "\n";
#ifdef __linux__
#elif _WIN32
		system("pause");
#else
#endif
	}
	else
	{
		//cout << "\n\nTraining sample index is " << training_block_index << " and seems to have been loaded correctly.";
#ifdef __linux__

#elif _WIN32
		//system("pause");
#else

#endif
	}
	file.close();
}
void read_weights_from_file()
{
	std::ifstream in(_relative_files_path + "/" + "model.hex", std::ios_base::binary);
	if (in.good())
	{
		for (int i = 0; i < numberOf_X; i++)
		{
			for (int k = 0; k < numberOf_H; k++)
			{
				in.read((char*)&W1[i][k], sizeof(float));
			}
		}
		for (int j = 0; j < numberOf_Y; j++)
		{
			for (int k = 0; k < numberOf_H; k++)
			{
				in.read((char*)&W2[k][j], sizeof(float));
			}
		}
		for (int k = 0; k < numberOf_H; k++)
		{
			in.read((char*)&hidden_bias[k], sizeof(float));
		}
		for (int j = 0; j < numberOf_Y; j++)
		{
			in.read((char*)&output_bias[j], sizeof(float));
		}
		in.read((char*)&_err_epoca_min_value, sizeof(float));
		in.read((char*)&_epsilon, sizeof(float));
		//in.read((char*)&x[numberOf_X - 1], sizeof(float));
		//in.read((char*)&h[numberOf_H - 1], sizeof(float));
	}
}
void write_weights_on_file(){
	std::ofstream fw(_relative_files_path + "/" + "model.hex", std::ios_base::binary);
	if (fw.good()){
		for (int i = 0; i < numberOf_X; i++){
			for (int k = 0; k < numberOf_H; k++){
				fw.write((char*)&W1[i][k], sizeof(float));
			}
		}
		for (int j = 0; j < numberOf_Y; j++){
			for (int k = 0; k < numberOf_H; k++){
				fw.write((char*)&W2[k][j], sizeof(float));
			}
		}
		for (int k = 0; k < numberOf_H; k++){
			fw.write((char*)&hidden_bias[k], sizeof(float));
		}
		for (int j = 0; j < numberOf_Y; j++){
			fw.write((char*)&output_bias[j], sizeof(float));
		}
		fw.write((char*)&_err_epoca_min_value, sizeof(float));
		fw.write((char*)&_epsilon, sizeof(float));
		//fw.write((char*)&x[numberOf_X - 1], sizeof(float));
		//fw.write((char*)&h[numberOf_H - 1], sizeof(float));
		fw.close();
		//cout << "\nFile closed... \n\n";
	}
	else
		cout << "Problem with opening file";
}
void normalizeArray(float* arr, float* normArr, int size) {
	float minVal = arr[0];
	float maxVal = arr[0];
	// Trova il minimo e il massimo
	for (int i = 1; i < size; i++) {
		if (arr[i] < minVal) minVal = arr[i];
		if (arr[i] > maxVal) maxVal = arr[i];
	}
	// Normalizza i valori
	for (int i = 0; i < size; i++) {
		if (maxVal != minVal) {
			normArr[i] = (arr[i] - minVal) / (maxVal - minVal);
		}
		else {
			normArr[i] = 0; // Evita divisione per zero nel caso di valori uguali
		}
	}
}
float mean_square_error(const float* arr1, const float* arr2, int size) {
	float sum = 0.00f;
	for (int i = 0; i < size; ++i) {
		float diff = arr1[i] - arr2[i];
		sum += (diff * diff);  // quadrato della differenza
	}
	// MSE = (1 / N) * Σ (diff^2)
	return sum / size;
}
float overallMean(const float* arr1, const float* arr2, int size) {
	float sum = 0.0f;
	// Sommiamo tutti gli elementi di entrambi gli array
	for (int i = 0; i < size; ++i) {
		sum += arr1[i] + arr2[i];
	}
	// La media complessiva è la somma divisa per il numero totale di elementi (2*size)
	return sum / (2 * size);
}
float calculateErrorPercentage(float mse, float overallMean) {
	// Calcola il Root Mean Squared Error (RMSE)
	float rms = sqrt(mse);
	// Calcola la percentuale: (RMSE / media complessiva) * 100
	float errorPercentage = (rms / overallMean) * 100.00f;
	return errorPercentage;
}
float calculateVariance(const float* data, int size) {
	// Calcolo della media
	float sum = 0.0f;
	for (int i = 0; i < size; ++i) {
		sum += data[i];
	}
	float mean = sum / size;
	// Calcolo della somma dei quadrati delle differenze
	float sumSquaredDifferences = 0.0f;
	for (int i = 0; i < size; ++i) {
		float diff = data[i] - mean;
		sumSquaredDifferences += diff * diff;
	}
	// La varianza (per popolazione) è la media dei quadrati delle differenze
	return sumSquaredDifferences / size;
}






