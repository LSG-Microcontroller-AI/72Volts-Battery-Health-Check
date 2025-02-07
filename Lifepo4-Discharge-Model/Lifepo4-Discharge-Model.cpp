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
#include "imgui.h"
#include "implot.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <vector>



#ifdef __linux__

#elif _WIN32
#include <conio.h>
#include <windows.h>
#include <ctime>

#else

#endif

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

float _err_rete = 0.00f;

float _err_amm = 0.00025f;

float _epsilon = 0.01;

const uint8_t numberOf_X = 2;

const uint8_t numberOf_H = 8;

const uint8_t numberOf_Y = 6;

float output_bias[numberOf_Y] = { 0.00 };

float hidden_bias[numberOf_H] = { 0.00 };

uint16_t const training_samples = 158;

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

GLFWwindow* window;

std::vector<double> ascissa;

std::vector<double> ordinata1;

std::vector<double> ordinata2;

void render_plot(const char* window_name) {

	ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_Always);
	
	ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);

	ImGui::Begin(window_name, nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);

	ImVec2 plot_size = ImVec2(ImGui::GetWindowSize().x - 20, ImGui::GetWindowSize().y - 60);

	if (ImPlot::BeginPlot("Andamento Errore", plot_size)) {
		
		// ✅ Cambia il colore della griglia in giallo acceso
		ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(1.0f, 1.0f, 0.3f, 0.7f));

		// ✅ Cambia il colore della linea in azzurro chiaro
		ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f, 0.7f, 1.0f, 1.0f));

		// ✅ Attiva i marker (punti) e li imposta in azzurro chiaro
		ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1.0f, ImVec4(0.3f, 0.8f, 1.0f, 1.0f), 1.0f);


		// Imposta gli assi con AutoFit per adattarli ai dati
		ImPlot::SetupAxes("Epoca", "Errore", ImPlotAxisFlags_AutoFit);
		
		ImPlot::SetupAxisLimits(ImAxis_X1, 0, (ascissa.empty() ? 100 : ascissa.back() + 10), ImGuiCond_Always);
		
		ImPlot::SetupAxisLimits(ImAxis_Y1, 0, (ordinata1.empty() ? 1.0 : *std::max_element(ordinata1.begin(), ordinata1.end()) + 0.1), ImGuiCond_Always);

		// Controlliamo se ci sono dati da disegnare
		if (!ascissa.empty()) {

			ImPlot::PlotLine("Errore", ascissa.data(), ordinata1.data(), ascissa.size());
		}

		// ✅ Ripristina i colori predefiniti (Griglia e Linea)
		ImPlot::PopStyleColor(2);

		ImPlot::EndPlot();
	}

	ImGui::End();
}

void render_plot2(const char* window_name) {
	ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_Always);

	ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);

	ImGui::Begin(window_name, nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);

	ImVec2 plot_size = ImVec2(ImGui::GetWindowSize().x - 20, ImGui::GetWindowSize().y - 60);

	if (ImPlot::BeginPlot("Andamento Errore", plot_size)) {
		// ✅ Cambia il colore della griglia in giallo acceso
		ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(1.0f, 1.0f, 0.3f, 0.7f));

		// ✅ Cambia il colore della linea in azzurro chiaro
		ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f, 0.7f, 1.0f, 1.0f));

		// ✅ Rende i punti più piccoli (1.5f per ridurli)
		ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1.5f, ImVec4(0.3f, 0.8f, 1.0f, 1.0f), 1.0f);

		// Imposta gli assi con AutoFit per adattarli ai dati
		ImPlot::SetupAxes("Epoca", "Errore", ImPlotAxisFlags_AutoFit);
		ImPlot::SetupAxisLimits(ImAxis_X1, 0, (ascissa.empty() ? 100 : ascissa.back() + 10), ImGuiCond_Always);
		ImPlot::SetupAxisLimits(ImAxis_Y1, 0, (ordinata2.empty() ? 1.0 : *std::max_element(ordinata2.begin(), ordinata2.end()) + 0.1), ImGuiCond_Always);

		// Controlliamo se ci sono dati da disegnare
		if (!ascissa.empty()) {
			ImPlot::PlotLine("Errore", ascissa.data(), ordinata2.data(), ascissa.size());
		}

		// ✅ Ripristina i colori predefiniti (Griglia e Linea)
		ImPlot::PopStyleColor(2);

		ImPlot::EndPlot();
	}

	ImGui::End();
}

GLFWwindow* InitWindow() {

	if (!glfwInit()) {
		return nullptr;
	}

	// Abilita l'hint per una finestra massimizzata
	glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

	// Crea la finestra con dimensioni standard
	GLFWwindow* window = glfwCreateWindow(1280, 720, "Grafici Seno e Coseno", NULL, NULL);
	if (!window) {
		
		glfwTerminate();
		
		return nullptr;
	}

	// Ora massimizziamo la finestra dopo la creazione
	glfwMaximizeWindow(window);

	glfwMakeContextCurrent(window);

	glfwSwapInterval(1);

	// Inizializzazione ImGui + ImPlot
	IMGUI_CHECKVERSION();

	ImGui::CreateContext();

	ImPlot::CreateContext();

	ImGui_ImplGlfw_InitForOpenGL(window, true);

	ImGui_ImplOpenGL3_Init("#version 130");

	return window;
}

void open_plots()
{
	glfwPollEvents();

	ImGui_ImplOpenGL3_NewFrame();

	ImGui_ImplGlfw_NewFrame();

	ImGui::NewFrame();

	render_plot("Errore vs err_rete");

	render_plot2("Errore vs err_epoca");

	ImGui::Render();

	int display_w, display_h;

	glfwGetFramebufferSize(window, &display_w, &display_h);

	glViewport(0, 0, display_w, display_h);

	glClear(GL_COLOR_BUFFER_BIT);

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(window);

	GLenum err;

	while ((err = glGetError()) != GL_NO_ERROR) {

		std::cout << "Errore OpenGL: " << err << std::endl;

	}
}

int main()
{
	window = InitWindow();

#ifdef __linux__

#elif _WIN32

	HWND consoleWindow = GetConsoleWindow();

	// Sposta e massimizza la finestra
	SetWindowPos(consoleWindow, nullptr, -1920, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER);

	ShowWindow(consoleWindow, SW_MAXIMIZE);

#else

#endif

	init();

#ifdef __linux__

	// no sound on linux

#elif _WIN32

	Beep(3000, 200);

#else

#endif

	char response;

	cout << "\n Do you want load the weights file\n";

#ifdef __linux__

	response = std::cin.get();

	std::cin.ignore();

#elif _WIN32

	response = _getch();

#else

#endif

	if (response == 'y')
	{
		cout << "\n Weights loaded\n";

		read_weights_from_file();
	}
	else
	{
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

	if (response == 'y')
	{
		cout << "\n Start to learning......\n";

		apprendi();
	}

	lavora();

	cout << "press a key..\n\n";
}

double xavier_init(double n_x, double n_y)
{
	return sqrt(6.0) / sqrt(n_x + n_y);
}

void init()
{
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

	for (int i = 0; i < (numberOf_X); i++)
	{
		//x[i] = 0.00f;
		cout << "x[" << i << "]" << "=" << x[i] << "\n";
	}

	for (int i = 0; i < numberOf_H; i++) {
		cout << "hidden_bias[" << i << "]" << "=" << hidden_bias[i] << "-BIAS" << "\n";
	}


	//-----------------------------------	console hidden values + output bias values

	for (int i = 0; i < (numberOf_H); i++)
	{
		//h[i] = 0.00f;
		cout << "h[" << i << "]" << "=" << h[i] << "\n";
	}

	for (int i = 0; i < numberOf_Y; i++) {
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
			W1[i][k] = get_random_number_from_xavier();

			cout << "W1[" << i << "]" << "[" << k << "]" << "=" << W1[i][k] << "\n";
		}
	}

	//-----------------------------------	console W2 values

	for (int k = 0; k < numberOf_H; k++)
	{
		for (int j = 0; j < numberOf_Y; j++)
		{
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

		x[1] = log(x[0] + 1.0f) / 10.0f;


		//// Operazioni sui valori
		//x[0] = x[0] / 100.00f;
		//x[1] = x[1] / 1000.00f;

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

float err_min_rete = FLT_MAX;

void apprendi()
{
	int epoca = 0;

	float err_epoca;

	int cout_counter = 0;

	auto start = std::chrono::system_clock::now();

	read_samples_from_file_diagram_battery();

	float err_epoca_min_value = FLT_MAX;

	do
	{
		err_epoca = 0.00f;

		for (unsigned long p = 0; p < training_samples; p++)
		{
			x[0] = log(amps_training[p] + 1.0f) / 10.0f;

			x[1] = log(watts_hour_training[p] + 1.0f) / 10.0f;

			for (int i = 0; i < numberOf_Y; i++) {

				d[i] = battery_out_training[p][i] / 10.00f;

			}

			forward();

			back_propagate();


			if (_err_rete > err_epoca)
			{
				//cout << "ciclo: " << p << "  errore_rete= " << _err_rete << "\n";
				err_epoca = _err_rete;
			}
		}


		if (err_epoca_min_value > err_epoca) {

			err_epoca_min_value = err_epoca;
		}

		epoca++;

		ascissa.push_back(epoca);

		ordinata1.push_back(_err_rete);

		ordinata2.push_back(err_epoca);

		open_plots();

		/*	std::cout << "\nepoca:" << epoca <<
				"\nlast modified date: " << global_time_recorded <<
				"\nerr_epoca=" << err_epoca <<
				" min.err_epoca= " << err_epoca_min_value <<
				" _err_rete=" << _err_rete <<
				" min._err_rete= " << err_min_rete <<
				"\n";*/

				//cout_counter = 0;

		if ((err_epoca < err_epoca_min_value) || (_err_rete < err_min_rete))
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

			if (localtime_s(&local_time, &now) != 0)
			{
				std::cerr << "Errore nella conversione del tempo.\n";
			}


			/*	global_time_recorded = std::to_string(local_time.tm_mday) + "/" +
					std::to_string(local_time.tm_mon + 1) + "/" +
					std::to_string(local_time.tm_year + 1900) + " " +
					std::to_string(local_time.tm_hour) + ":" +
					std::to_string(local_time.tm_min) + ":" +
					std::to_string(local_time.tm_sec);*/


#else

#endif

			err_min_rete = _err_rete;
			//std::cout << "\nwrite on file\n";
			write_weights_on_file();
		}
		//}

	} while (err_epoca > _err_amm);

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
#else
#endif



#ifdef __linux__
	getchar();
#elif _WIN32
	_getch();
#else

#endif
}

void forward()
{
	for (int k = 0; k < (numberOf_H); k++)
	{
		float Zk = 0.00f;

		for (int i = 0; i < numberOf_X; i++)
		{
			Zk += (W1[i][k] * x[i]);
		}

		//insert X bias
		Zk += hidden_bias[k];

		h[k] = sigmoid_activation(Zk);
	}

	for (int j = 0; j < numberOf_Y; j++)
	{
		float Zj = 0.00f;

		for (int k = 0; k < numberOf_H; k++)
		{
			Zj += (W2[k][j] * h[k]);
		}

		//insert H bias
		Zj += output_bias[j];

		y[j] = sigmoid_activation(Zj);
	}
}

void back_propagate()
{
	float err_H[numberOf_H] = { 0.00f };

	float delta = 0.00f;

	_err_rete = 0.00f;

	for (int j = 0; j < numberOf_Y; j++)
	{
		if (abs(d[j] - y[j]) > _err_rete)
		{
			_err_rete = abs(d[j] - y[j]);
		}

		delta = (d[j] - y[j]) * y[j] * (1.00f - y[j]);

		for (int k = 0; k < numberOf_H; k++)
		{
			W2[k][j] += (_epsilon * delta * h[k]);

			err_H[k] += (delta * W2[k][j]);
		}

		// Aggiornamento del bias del livello di uscita
		output_bias[j] += _epsilon * delta;
	}

	for (int k = 0; k < numberOf_H; k++)
	{
		delta = err_H[k] * h[k] * (1.00f - h[k]);

		for (int i = 0; i < numberOf_X; i++)
		{
			W1[i][k] += (_epsilon * delta * x[i]);
		}

		hidden_bias[k] += _epsilon * delta;
	}
}

double get_random_number_from_xavier()
{
	uniform_real_distribution<double> distribution(_lower_bound_xavier, _upper_bound_xavier);

	double random_value = distribution(generator);

	return random_value;
}

float sigmoid_activation(float Z)
{
	//return 1.00f / (1.00f + pow(M_E, -Z));
	return 1.00f / (1.00f + exp(-Z));
}

void read_samples_from_file_diagram_battery()
{
	//std::cout << "Directory corrente: " << std::filesystem::current_path() << std::endl;

	std::string filename = _relative_files_path + "/" + "72V_Battery.CSV";

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
			try
			{
				std::getline(file, line);

				ss1.str(line);

				std::getline(ss1, item, ';');

				std::getline(ss1, item, ';');

				std::getline(ss1, item, ';');

				battery_out_training[training_block_index][training_row_pre_index] = std::stod(item);

				cout << "battery[" << training_block_index << "]" << "[" << training_row_pre_index << "] = " << battery_out_training[training_block_index][training_row_pre_index] << "\n";
			}
			catch (...) {};

			break;
		case 6:
			try
			{
				std::getline(file, line);

				ss1.str(line);

				std::getline(ss1, item, ';');

				std::getline(ss1, item, ';');

				std::getline(ss1, item, ';');

				watts_hour_training[training_block_index] = std::stod(item);

				cout << "Watts/hour[" << training_block_index << "] = " << watts_hour_training[training_block_index] << "\n";
			}
			catch (...) {};

			break;
		case 7:
			try
			{
				std::getline(file, line);

				ss1.str(line);

				std::getline(ss1, item, ';');

				std::getline(ss1, item, ';');

				std::getline(ss1, item, ';');

				amps_training[training_block_index] = std::stod(item);

				cout << "Ampere[" << training_block_index << "] = " << amps_training[training_block_index] << "\n";
			}
			catch (...) {};

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

	if (training_block_index != training_samples)
	{
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

void read_weights_from_file()
{
	std::ifstream in(_relative_files_path + "/" + "model.bin", std::ios_base::binary);

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

		//in.read((char*)&x[numberOf_X - 1], sizeof(float));

		//in.read((char*)&h[numberOf_H - 1], sizeof(float));
	}
}

void write_weights_on_file()
{
	try
	{
		//cout << "\nWriting to file... \n\n";

		std::ofstream fw(_relative_files_path + "/" + "model.bin", std::ios_base::binary);

		if (fw.good())
		{
			for (int i = 0; i < numberOf_X; i++)
			{
				for (int k = 0; k < numberOf_H; k++)
				{
					fw.write((char*)&W1[i][k], sizeof(float));
				}
			}

			for (int j = 0; j < numberOf_Y; j++)
			{
				for (int k = 0; k < numberOf_H; k++)
				{
					fw.write((char*)&W2[k][j], sizeof(float));
				}
			}


			for (int k = 0; k < numberOf_H; k++)
			{
				fw.write((char*)&hidden_bias[k], sizeof(float));
			}

			for (int j = 0; j < numberOf_Y; j++)
			{
				fw.write((char*)&output_bias[j], sizeof(float));
			}


			//fw.write((char*)&x[numberOf_X - 1], sizeof(float));

			//fw.write((char*)&h[numberOf_H - 1], sizeof(float));

			fw.close();

			//cout << "\nFile closed... \n\n";
		}
		else
			cout << "Problem with opening file";
	}
	catch (const char* msg)
	{
		cerr << msg << endl;
	}
}





