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

/* Neural Network Class */
class NeuralNetwork {
   public:
      /* Neural Network Constructor */
      NeuralNetwork(int inputs, vector<int> hidden_layers, int outputs);

      /* Print Number of Neurons in Each Layer */
      void printNetworkSetup();

      /* Print Weights of Entire Network */
      void printWB();

      /* Calculate Network Output (Feedforward) */
      vector<float> calculate(vector<float> inputs);

   private:
      /* Number of Network Inputs */
      int num_inputs;

      /* Number of Network Outputs */
      int num_outputs;

      /* Number of Hidden Layers */
      int num_hidden_layers;

      /* Training Step */
      int training_step;

      /* Contains Neurons of Hidden Layers */
      vector< vector<Neuron> > hidden_layers;

      /* Contains Input Neurons */
      vector<Neuron> input_layer;

      /* Contains Output Neurons */
      vector<Neuron> output_layer;

      /* Feed Inputs Into Network */
      void setInputs(vector<float> inputs);

      /* Calculate Outputs of Hidden Neurons */
      void calculateHidden();

      /* Calculate Outputs of Network */
      vector<float> calculateOutputs();
};

#endif