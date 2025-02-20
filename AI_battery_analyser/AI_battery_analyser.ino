/*
 Name:		AI_battery_analyser.ino
 Created:	2/18/2025 12:52:20 PM
 Author:	luigi.santagada
*/
#include <Arduino.h>
#ifndef RAMSTART
extern int __data_start;
#endif
extern int __bss_end;
extern void* __brkval;
#include <EEPROM.h>
const uint8_t numberOf_X = 2;
const uint8_t numberOf_H = 25;
const uint8_t numberOf_Y = 6;
float y_normalized_real_data[numberOf_Y] = { 0.00 };
float y_normalized_model_data[numberOf_Y] = { 0.00 };
float output_bias[numberOf_Y] = { 0.00 };
float hidden_bias[numberOf_H] = { 0.00 };
float W1[numberOf_X][numberOf_H] = { 0.00 };
float W2[numberOf_H][numberOf_Y] = { 0.00 };
float x[numberOf_X] = { 0.00 };
float h[numberOf_H] = { 0.00 };
float y[numberOf_Y] = { 0.00f };
float processed_data[6] = { 0.00 };
void read_weights_from_eeprom();
// the setup function runs once when you press reset or power the board
void setup() {
	Serial.begin(9600);
	read_weights_from_eeprom();
	/*x[0] = 23.00f;
	x[1] = 246.00f;*/
	processed_data[0] = 1.75f;
	processed_data[1] = 1.34f;
	processed_data[2] = 1.99f;
	processed_data[3] = 1.90f;
	processed_data[4] = 1.85f;
	processed_data[5] = 1.93f;
}
// the loop function runs over and over again until power down or reset
void loop() {
	int ram_libera = freeMemory();
	Serial.print(F("RAM libera: "));
	Serial.print(ram_libera);
	Serial.println(F(" byte"));
	x[0] = 23.00f;
	x[1] = 246.00f;
	x[0] = log(x[0] + 1.0f) / 10.0f;
	x[1] = log(x[1] + 1.0f) / 10.0f;
	forward();
	print_model_data();
	normalizeArray(processed_data, y_normalized_real_data, numberOf_Y);
	normalizeArray(y, y_normalized_model_data, numberOf_Y);
	float mse = meanSquaredError(y_normalized_real_data, y_normalized_model_data, numberOf_Y);
	float overall_mean = overallMean(y_normalized_real_data, y_normalized_model_data,numberOf_Y);
	uint8_t percentage = calculateErrorPercentage(mse, overall_mean);
	Serial.println(percentage);
	Serial.println(mse, 10);
	print_normalizer_processed_data();
	print_normalizer_model_data();
	delay(2000);
}
void print_model_data(){
	Serial.println();
	Serial.print(F("x[0] = ")); Serial.print(exp(x[0] * 10.00f)); Serial.print(F("    x[1] = ")); Serial.println(exp(x[1] * 10.00f));
	Serial.print(F("\n\ny[0] = ")); Serial.println(y[0] * 10.00f);
	Serial.print(F("y[1] = ")); Serial.println(y[1] * 10.00f);
	Serial.print(F("y[2] = ")); Serial.println(y[2] * 10.00f);
	Serial.print(F("y[3] = ")); Serial.println(y[3] * 10.00f);
	Serial.print(F("y[4] = ")); Serial.println(y[4] * 10.00f);
	Serial.print(F("y[5] = ")); Serial.println(y[5] * 10.00f);
	delay(2000);
}
void print_normalizer_processed_data() {
	Serial.println();
	Serial.print(F("y_normalized[0] = ")); Serial.println(y_normalized_real_data[0]);
	Serial.print(F("y_normalized[1] = "));  Serial.println(y_normalized_real_data[1]);
	Serial.print(F("y_normalized[2] = "));  Serial.println(y_normalized_real_data[2]);
	Serial.print(F("y_normalized[3] = "));  Serial.println(y_normalized_real_data[3]);
	Serial.print(F("y_normalized[4] = "));  Serial.println(y_normalized_real_data[4]);
	Serial.print(F("y_normalized[5] = "));  Serial.println(y_normalized_real_data[5]);
	delay(2000);
}
void print_normalizer_model_data() {
	Serial.println();
	Serial.print(F("y_normalized_model_data[0] = ")); Serial.println(y_normalized_model_data[0]);
	Serial.print(F("y_normalized_model_data[1] = "));  Serial.println(y_normalized_model_data[1]);
	Serial.print(F("y_normalized_model_data[2] = "));  Serial.println(y_normalized_model_data[2]);
	Serial.print(F("y_normalized_model_data[3] = "));  Serial.println(y_normalized_model_data[3]);
	Serial.print(F("y_normalized_model_data[4] = "));  Serial.println(y_normalized_model_data[4]);
	Serial.print(F("y_normalized_model_data[5] = "));  Serial.println(y_normalized_model_data[5]);
	delay(2000);
}
float relu(float x) {
	return (x > 0) ? x : 0;
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
void read_weights_from_eeprom() {
	int addr = 0;  // partiamo dall'indirizzo 0 della EEPROM
	// Legge la matrice W1: dimensione [numberOf_X][numberOf_H]
	for (int i = 0; i < numberOf_X; i++) {
		for (int k = 0; k < numberOf_H; k++) {
			EEPROM.get(addr, W1[i][k]);
			addr += sizeof(float);
		}
	}
	// Legge la matrice W2: dimensione [numberOf_H][numberOf_Y]
	for (int j = 0; j < numberOf_Y; j++) {
		for (int k = 0; k < numberOf_H; k++) {
			EEPROM.get(addr, W2[k][j]);
			addr += sizeof(float);
		}
	}
	// Legge il vettore dei bias per il layer nascosto
	for (int k = 0; k < numberOf_H; k++) {
		EEPROM.get(addr, hidden_bias[k]);
		addr += sizeof(float);
	}
	// Legge il vettore dei bias per l'output
	for (int j = 0; j < numberOf_Y; j++) {
		EEPROM.get(addr, output_bias[j]);
		addr += sizeof(float);
	}
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
float meanSquaredError(const float* arr1, const float* arr2, int size) {
	float sum = 0.0f;
	for (int i = 0; i < size; ++i) {
		float diff = arr1[i] - arr2[i];
		sum += diff * diff;  // quadrato della differenza
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
uint8_t calculateErrorPercentage(float mse, float overallMean) {
	// Calcola il Root Mean Squared Error (RMSE)
	float rms = sqrt(mse);
	// Calcola la percentuale: (RMSE / media complessiva) * 100
	uint8_t errorPercentage = (rms / overallMean) * 100.0f;
	return errorPercentage;
}
int freeMemory() {
	int free_memory;
	if ((int)__brkval == 0) {
		free_memory = ((int)&free_memory) - ((int)&__bss_end);
	}
	else {
		free_memory = ((int)&free_memory) - ((int)__brkval);
	}
	return free_memory;
}
