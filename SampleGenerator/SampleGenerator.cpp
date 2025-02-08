#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>  // C++17

using namespace std;
namespace fs = std::filesystem;

int main() {
    // Verifica se la directory "Bin" esiste, altrimenti la crea
    if (!fs::exists("Bin")) {
        if (!fs::create_directory("Bin")) {
            cerr << "Impossibile creare la directory Bin." << endl;
            return 1;
        }
    }

    // Valori iniziali per le batterie (dal tuo esempio)
    double initValues[6] = { 3.00, 3.11, 3.12, 3.07, 3.11, 3.06 };
    const int nGroups = 100;  // numero di gruppi
    // Il valore finale desiderato per ciascuna batteria è 2.50

    // Apro il file "output.csv" nella cartella Bin in modalità scrittura
    ofstream outFile("Bin/72V_Battery.CSV");
    if (!outFile) {
        cerr << "Errore nell'apertura del file per la scrittura." << endl;
        return 1;
    }

    // Genera 100 gruppi
    for (int i = 1; i <= nGroups; i++) {
        // Genera le 6 righe dei dati delle batterie (B0-B5)
        for (int j = 0; j < 6; j++) {
            // Calcolo lineare: per i = 1, valore = initValues[j]; per i = nGroups, valore = 2.50
            double v = initValues[j] + (2.50 - initValues[j]) * (i - 1) / (nGroups - 1);
            // Scrive la riga nel formato: "0;B{j};{v};0;{v}"
            outFile << "0;B" << j << ";"
                << fixed << setprecision(2) << v
                << ";0;" << fixed << setprecision(2) << v << "\n";
        }
        // Riga dei Watt: la terza colonna aumenta di 10 per gruppo
        int watts = (i - 1) * 10;
        outFile << "watts;;" << watts << ";;\n";

        // Riga degli Ampere: aumenta anch'esso di 10 per gruppo
        int amps = (i - 1) * 1;
        outFile << "amps;;" << amps << ";;\n";
    }

    outFile.close();
    cout << "File generato correttamente in Bin/72V_Battery.CSV" << endl;
    return 0;
}
