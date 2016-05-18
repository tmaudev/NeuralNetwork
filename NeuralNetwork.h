/** Artificial Neural Network Library
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#ifndef NEURAL_NETWORK_H

#include <iostream>
#include <vector>

#include "Neuron.h"

using namespace std;

class NeuralNetwork {
   public:
      NeuralNetwork(int inputs, vector<int> hidden_layers, int outputs);
      void printNetworkSetup();
      void printWeights();
      vector<float> calculate(vector<float> inputs);

   private:
      int num_inputs;
      int num_outputs;
      int num_hidden_layers;
      vector< vector<Neuron> > hidden_layers;
      vector<Neuron> input_layer;
      vector<Neuron> output_layer;

      void setInputs(vector<float> inputs);
      void calculateHidden();
      vector<float> calculateOutputs();
};

#endif