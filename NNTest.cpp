/** Artificial Neural Network Library Test
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include "NeuralNetwork.h"
#define HIDDEN_LAYERS 1

int main() {
   vector<int> hidden_layers(HIDDEN_LAYERS);
   vector<float> inputs;
   vector<float> outputs;

   hidden_layers[0] = 5;

   NeuralNetwork net(1, hidden_layers, 1);

   net.printNetworkSetup();
   net.printWB();

   inputs.push_back(5);

   outputs = net.calculate(inputs);

   printf("Output: %f\n", outputs[0]);

   return 0;
}