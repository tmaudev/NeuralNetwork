/** Artificial Neural Network Library Test
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"
#define HIDDEN_LAYERS 1

/* Hyperbolic Tangent Activation Function */
float hyperbolic_tangent(float input, float abs_max, float slope) {
   float output = 0;
   float exp_pos = exp(slope * input);
   float exp_neg = exp(-slope * input);

   output = abs_max * ((exp_pos - exp_neg) / (exp_pos + exp_neg));

   return output;
}

int main() {
   vector<int> hidden_layers(HIDDEN_LAYERS);
   vector<float> inputs;
   vector<float> outputs;
   vector< vector<float> > training_input;
   vector< vector<float> > training_output;
   vector<float> *input;
   vector<float> *output;
   int temp;
   float p[] = {1500,
                1510,
                1520,
                1530,
                1540,
                1550,
                1500,
                1510,
                1520,
                1530,
                1540,
                1550,
                1560,
                1570,
                1580,
                1590,
                1600,
                1610,
                1620,
                1630,
                1640,
                1650,
                1660,
                1670,
                1680,
                1690,
                1700,
                1710,
                1720,
                1730,
                1740,
                1750,
                1760,
                1770,
                1780,
                1790,
                1800,
                1810,
                1820,
                1830,
                1840,
                1850,
                1860,
                1870,
                1880,
                1890,
                1900,
                1890,
                1880,
                1870,
                1860,
                1850,
                1840,
                1830,
                1820,
                1810,
                1800,
                1790,
                1780,
                1770,
                1760,
                1750,
                1740,
                1730,
                1720,
                1710,
                1700,
                1690,
                1680,
                1670,
                1660,
                1650,
                1640,
                1630,
                1620,
                1610,
                1600,
                1590,
                1580,
                1570,
                1560,
                1550,
                1540,
                1530,
                1520,
                1510,
                1500,
                1490,
                1480,
                1470,
                1460,
                1450,
                1440,
                1430,
                1420,
                1410,
                1400,
                1390,
                1380,
                1370,
                1360,
                1350,
                1340,
                1330,
                1320,
                1310,
                1300,
                1290,
                1280,
                1270,
                1260,
                1250,
                1240,
                1230,
                1220,
                1210,
                1200,
                1190,
                1180,
                1170,
                1160,
                1150,
                1140,
                1130,
                1120,
                1110,
                1100,
                1110,
                1120,
                1130,
                1140,
                1150,
                1160,
                1170,
                1180,
                1190,
                1200,
                1210,
                1220,
                1230,
                1240,
                1250,
                1260,
                1270,
                1280,
                1290,
                1300,
                1310,
                1320,
                1330,
                1340,
                1350,
                1360,
                1370,
                1380,
                1390,
                1400,
                1410,
                1420,
                1430,
                1440,
                1450,
                1460,
                1470,
                1480,
                1490,
                1500,
                1450,
                1460,
                1470,
                1480,
                1490,
                1500};

   float t[] = {-0.04,
                -0.04,
                -0.04,
                0.02,
                0.08,
                0.14,
                -0.04,
                -0.04,
                -0.04,
                0.02,
                0.08,
                0.14,
                0.23,
                0.32,
                0.43,
                0.56,
                0.68,
                0.81,
                0.95,
                1.06,
                1.2,
                1.35,
                1.51,
                1.62,
                1.77,
                1.94,
                2.12,
                2.33,
                2.49,
                2.61,
                2.76,
                2.87,
                3.05,
                3.28,
                3.47,
                3.64,
                3.8,
                4.01,
                4.31,
                4.5,
                4.62,
                4.8,
                4.98,
                5.11,
                5.32,
                5.47,
                5.54,
                5.56,
                5.52,
                5.27,
                5.03,
                4.8,
                4.61,
                4.51,
                4.22,
                4.06,
                3.82,
                3.71,
                3.61,
                3.41,
                3.34,
                3.01,
                2.89,
                2.74,
                2.55,
                2.36,
                2.21,
                2.04,
                1.9,
                1.69,
                1.55,
                1.43,
                1.3,
                1.14,
                1.01,
                0.87,
                0.79,
                0.68,
                0.56,
                0.46,
                0.38,
                0.28,
                0.21,
                0.15,
                0.08,
                0.06,
                0.06,
                0.05,
                0.05,
                0.02,
                -0.03,
                -0.07,
                -0.11,
                -0.16,
                -0.23,
                -0.31,
                -0.37,
                -0.44,
                -0.51,
                -0.6,
                -0.72,
                -0.81,
                -0.88,
                -0.94,
                -1.04,
                -1.17,
                -1.28,
                -1.37,
                -1.47,
                -1.55,
                -1.66,
                -1.74,
                -1.88,
                -2.04,
                -2.18,
                -2.29,
                -2.43,
                -2.56,
                -2.66,
                -2.77,
                -2.91,
                -3.08,
                -3.17,
                -3.36,
                -3.47,
                -3.61,
                -3.89,
                -3.81,
                -3.65,
                -3.46,
                -3.28,
                -3.18,
                -2.99,
                -2.85,
                -2.7,
                -2.52,
                -2.42,
                -2.34,
                -2.21,
                -2.07,
                -1.97,
                -1.85,
                -1.71,
                -1.58,
                -1.46,
                -1.38,
                -1.23,
                -1.1,
                -0.99,
                -0.94,
                -0.89,
                -0.81,
                -0.72,
                -0.62,
                -0.54,
                -0.45,
                -0.39,
                -0.33,
                -0.27,
                -0.2,
                -0.15,
                -0.1,
                -0.06,
                -0.04,
                0,
                0,
                0,
                -0.1,
                -0.06,
                -0.04,
                0,
                0,
                0};

   int p_size = sizeof(p) / sizeof(*p);

   /* Create Sample Training Input and Output */
   for (int i = 0; i < p_size; i++) {
      //temp = (rand() / (float)RAND_MAX) * 5 + 1;

      input = new vector<float>();
      output = new vector<float>();

      //cout << temp << endl;
      (*input).push_back(p[i] / 100.0);
      (*output).push_back(t[i]);

      training_input.push_back(*input);
      training_output.push_back(*output);
      //cout << temp * temp << endl;
   }

   // input.push_back(5);
   // output.push_back(11);
   // training_input.push_back(input);
   // training_output.push_back(output);

   hidden_layers[0] = 15;

   NeuralNetwork net(1, hidden_layers, 1);

   net.printNetworkSetup();
   //net.printWB();

   //cout << "Test: " << hyperbolic_tangent(0.7, 1, 1) << endl;
   //cout << "Compare: " << tanh(0.7);

   // outputs = net.calculate(input);

   // printf("\nOutput Before Training: %f\n", outputs[0]);

   cout << "Before Training:" << endl;
   for (int i = 0; i < 10; i++) {
      vector<float> test_input, test_output;
      test_input.clear();
      test_input.push_back(i + 1);

      test_output = net.calculate(test_input);

      cout << "Input: " << test_input[0] << "   Output: " << test_output[0] << endl;
   }

   net.train(0.00001, 10000000, training_input, training_output);

   //outputs = net.calculate(input);

   ofstream myfile;
   myfile.open("output.txt");

   cout << "\nAfter Training:" << endl;
   for (float i = 1100; i < 1910; i += 10) {
      vector<float> test_input, test_output;
      test_input.clear();
      test_input.push_back(i / 100.0);

      test_output = net.calculate(test_input);

      cout << "Input: " << test_input[0] * 100 << "   Output: " << test_output[0] << endl;
      myfile << test_output[0] << endl;
   }

   // cout << "\nAfter Training:" << endl;
   // for (float i = 0; i < 5; i++) {
   //    vector<float> test_input, test_output;
   //    test_input.clear();
   //    test_input.push_back(i + 1);

   //    test_output = net.calculate(test_input);

   //    cout << "Input: " << test_input[0] << "   Output: " << test_output[0] << endl;
   //    //myfile << test_output[0] << endl;
   // }

   myfile.close();

   //printf("Output After Training: %f\n\n", outputs[0]);

   //net.printWB();

   return 0;
}