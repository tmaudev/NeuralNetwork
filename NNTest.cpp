/** Artificial Neural Network Library Test
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include "NeuralNetwork.h"
#define HIDDEN_LAYERS 2

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

   hidden_layers[0] = 3;
   hidden_layers[1] = 5;

   NeuralNetwork net(1, hidden_layers, 1);

   net.printNetworkSetup();
   net.printWB();

   inputs.push_back(5);

   //cout << "Test: " << hyperbolic_tangent(0.7, 1, 1) << endl;

   outputs = net.calculate(inputs);

   printf("\nOutput: %f\n", outputs[0]);

   return 0;
}