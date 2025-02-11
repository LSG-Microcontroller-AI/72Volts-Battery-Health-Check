#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>    // per round()
using namespace std;

int main() {
    // Nome del file di input (modifica se necessario)
    const string inputFileName = "72V_Battery.CSV";
    ifstream infile(inputFileName);
    if (!infile.is_open()) {
        cerr << "Errore nell'apertura del file " << inputFileName << endl;
        return 1;
    }

    // Mappa per raggruppare i gruppi per valore intero di "amps"
    // La chiave è l'intero (da 1 a 50) e il valore è un vettore di gruppi;
    // ogni gruppo è rappresentato come un vettore di stringhe (8 righe)
    map<int, vector<vector<string>>> groupsByBattery;

    vector<string> group;  // contiene le 8 righe di un gruppo
    string line;
    int lineCount = 0;

    // Leggiamo il file riga per riga
    while (getline(infile, line)) {
        group.push_back(line);
        lineCount++;

        // Ogni gruppo è formato da 8 righe
        if (lineCount % 8 == 0) {
            // L'ottava riga contiene la riga "amps"
            string ampsLine = group[7];

            // Dividiamo la riga in token usando il ';' come separatore
            vector<string> tokens;
            istringstream ss(ampsLine);
            string token;
            while (getline(ss, token, ';')) {
                tokens.push_back(token);
            }

            double ampsValue = 0.0;
            // Presupponiamo che il valore degli amps sia il terzo token (indice 2)
            if (tokens.size() >= 3) {
                try {
                    ampsValue = stod(tokens[2]);
                }
                catch (const exception& e) {
                    cerr << "Errore nella conversione del valore amps: " << tokens[2] << endl;
                }
            }

            // Convertiamo il valore a intero, utilizzando round() (ad esempio 8.27 diventa 8)
            int bucket = static_cast<int>(round(ampsValue));

            // Consideriamo solo i gruppi con bucket da 1 a 50
            if (bucket >= 1 && bucket <= 50) {
                groupsByBattery[bucket].push_back(group);
            }

            // Ripuliamo il vettore per il prossimo gruppo
            group.clear();
        }
    }
    infile.close();

    // Per ciascun bucket (da 1 a 50) scriviamo i gruppi corrispondenti in un file separato
    for (int i = 1; i <= 50; i++) {
        // Se esistono gruppi per questo valore di amps
        if (groupsByBattery.find(i) != groupsByBattery.end()) {
            // Creiamo il nome del file, es. "Battery1.txt", "Battery2.txt", ecc.
            string outFileName = "Battery" + to_string(i) + ".txt";
            ofstream outfile(outFileName);
            if (!outfile.is_open()) {
                cerr << "Errore nell'apertura del file di output " << outFileName << endl;
                continue;
            }

            // Scriviamo tutti i gruppi che corrispondono a questo bucket
            for (const auto& grp : groupsByBattery[i]) {
                for (const auto& l : grp) {
                    outfile << l << "\n";
                }
                // Inseriamo una riga vuota tra un gruppo e l'altro (facoltativo)
                outfile << "\n";
            }
            outfile.close();
            cout << "Creato il file " << outFileName << " con "
                << groupsByBattery[i].size() << " gruppi." << endl;
        }
    }

    return 0;
}
