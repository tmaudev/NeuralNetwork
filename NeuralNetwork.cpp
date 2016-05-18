/** Artificial Neural Network Library
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int inputs, vector<int> hidden_config, int outputs) {
   vector<Neuron> layer;

   /* Save Network Configuration */
   num_hidden_layers = hidden_config.size();
   num_inputs = inputs;
   num_outputs = outputs;

   /* Create Vector of Input Nodes */
   for (int i = 0; i < num_inputs; i++) {
      input_layer.push_back(Neuron());
   }

   /* Create Vector of Hidden Layers */
   for (int i = 0; i < num_hidden_layers; i++) {
      layer.clear();

      /* Create Vector of Nodes for Each Layer */
      for (int j = 0; j < hidden_config[i]; j++) {
         if (i) {
            layer.push_back(Neuron(hidden_layers[i - 1], HT_ACTIVATION));
         }
         else {
            layer.push_back(Neuron(input_layer, HT_ACTIVATION));
         }
      }

      hidden_layers.push_back(layer);
   }

   /* Create Output Layer */
   for (int i = 0; i < outputs; i++) {
      output_layer.push_back(Neuron(hidden_layers[num_hidden_layers - 1], LINEAR_ACTIVATION));
   }
}

void NeuralNetwork::setInputs(vector<float> inputs) {
   for (int i = 0; i < input_layer.size(); i++) {
      input_layer[i].setOutput(inputs[i]);
   }
}

void NeuralNetwork::calculateHidden() {
   for (int i = 0; i < hidden_layers.size(); i++) {
      for (int j = 0; j < hidden_layers[i].size(); j++) {
         hidden_layers[i][j].calculate();
      }
   }
}

vector<float> NeuralNetwork::calculateOutputs() {
   vector<float> output;

   for (int i = 0; i < output_layer.size(); i++) {
      output_layer[i].calculate();
      output.push_back(output_layer[i].getOutput());
   }

   return output;
}

vector<float> NeuralNetwork::calculate(vector<float> inputs) {
   setInputs(inputs);
   calculateHidden();
   return calculateOutputs();
}

void NeuralNetwork::printNetworkSetup() {
   cout << "Layer Configuration: " << num_inputs;

   for (int i = 0; i < num_hidden_layers; i++) {
      cout << " " << hidden_layers[i].size();
   }

   cout << " " << num_outputs << endl;
}

void NeuralNetwork::printWeights() {
   vector<float> weights;
   float bias;

   for (int i = 0; i < num_hidden_layers; i++) {
      cout << "Layer " << i << ":" << endl;

      for (int j = 0; j < hidden_layers[i].size(); j++) {
         cout << "   Neuron " << j << ":";

         weights = hidden_layers[i][j].getWeights();
         for (int k = 0; k < weights.size(); k++) {
            printf(" %4.3f", weights[k]);
         }

         bias = hidden_layers[i][j].getBias();
         printf("   Bias: %4.3f\n", bias);
      }
      cout << endl;
   }

   cout << "Output Layer:" << endl;
   for (int i = 0; i < num_outputs; i++) {
      cout << "   Neuron " << i << ":";

      weights = output_layer[i].getWeights();
      for (int j = 0; j < weights.size(); j++) {
         printf(" %4.3f", weights[j]);
      }

      bias = output_layer[i].getBias();
      printf("   Bias: %4.3f\n", bias);
   }
}