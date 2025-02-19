/*
 Name:		AI_battery_analyser.ino
 Created:	2/18/2025 12:52:20 PM
 Author:	luigi.santagada
*/

#include <EEPROM.h>
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
// the setup function runs once when you press reset or power the board
void setup() {
	Serial.begin(9600);
	pinMode(8, OUTPUT);
	digitalWrite(8, LOW);
	read_weights_from_eeprom();
	delay(500);
	x[0] = 30.00f;
	x[1] = 500.00f;
	x[0] = log(x[0] + 1.0f) / 10.0f;
	x[1] = log(x[1] + 1.0f) / 10.0f;
	forward();
	Serial.println();
	Serial.print("x[0] = ");Serial.print(exp(x[0] * 10.00f));Serial.print("    x[1] = ");Serial.println(exp(x[1] * 10.00f));
	Serial.print("\n\ny[0] = ");Serial.println(y[0] * 10.00f);
	Serial.print("y[1] = ");Serial.println(y[1] * 10.00f);
	Serial.print("y[2] = ");Serial.println(y[2] * 10.00f);
	Serial.print("y[3] = ");Serial.println(y[3] * 10.00f);
	Serial.print("y[4] = ");Serial.println(y[4] * 10.00f);
	Serial.print("y[5] = ");Serial.println(y[5] * 10.00f);
	//digitalWrite(8, HIGH);
}
// the loop function runs over and over again until power down or reset
void loop() {
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
