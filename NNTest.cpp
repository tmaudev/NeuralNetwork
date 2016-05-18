/** Artificial Neural Network Library Test
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include "NeuralNetwork.h"
#define HIDDEN_LAYERS 2

int main() {
  vector<int> hidden_layers(HIDDEN_LAYERS);
  vector<float> inputs;
  vector<float> outputs;

  hidden_layers[0] = 5;
  hidden_layers[1] = 6;

  NeuralNetwork net(1, hidden_layers, 1);

  net.printNetworkSetup();
  net.printWeights();

  inputs.push_back(3);

  outputs = net.calculate(inputs);

  printf("Output: %f\n", outputs[0]);

  return 0;
}