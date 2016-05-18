/** Artificial Neural Network Library
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include "NeuralNetwork.h"

/* Neural Network Constructor */
NeuralNetwork::NeuralNetwork(int inputs, vector<int> hidden_config, int outputs) {
   vector<Neuron> layer;

   //THROW ERRORS IF WRONG INPUTS

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
         /* Previous Layer is a Hidden Layer */
         if (i) {
            layer.push_back(Neuron(hidden_layers[i - 1], HT_ACTIVATION));
         }
         /* Previous Layer is Input Layer */
         else {
            layer.push_back(Neuron(input_layer, HT_ACTIVATION));
         }
      }
      hidden_layers.push_back(layer);
   }

   /* Create Output Layer (Linear Activation Function) */
   for (int i = 0; i < outputs; i++) {
      output_layer.push_back(Neuron(hidden_layers[num_hidden_layers - 1], LINEAR_ACTIVATION));
   }
}

/* Print Number of Neurons in Each Layer */
void NeuralNetwork::printNetworkSetup() {
   /* Print Size of Input Layer */
   cout << "Layer Configuration: " << num_inputs;

   /* Print Size of Each Hidden Layer */
   for (int i = 0; i < num_hidden_layers; i++) {
      cout << " " << hidden_layers[i].size();
   }

   /* Print Size of Output Layer */
   cout << " " << num_outputs << endl;
}

/* Print Weights and Biases of Entire Network */
void NeuralNetwork::printWB() {
   vector<float> weights;
   float bias;

   /* Iterate Through Hidden Layers */
   for (int i = 0; i < num_hidden_layers; i++) {
      cout << "Layer " << i << ":" << endl;

      /* Iterate Through Each Neuron */
      for (int j = 0; j < hidden_layers[i].size(); j++) {
         cout << "   Neuron " << j << ":";

         /* Get Weights from Neuron */
         weights = hidden_layers[i][j].getWeights();

         /* Print Weights */
         for (int k = 0; k < weights.size(); k++) {
            printf(" %4.3f", weights[k]);
         }

         /* Get Bias from Neuron */
         bias = hidden_layers[i][j].getBias();

         /* Print Bias */
         printf("   Bias: %4.3f\n", bias);
      }
      cout << endl;
   }

   cout << "Output Layer:" << endl;

   /* Iterate Through Output Layer */
   for (int i = 0; i < num_outputs; i++) {
      cout << "   Neuron " << i << ":";

      /* Get Weights from Neuron */
      weights = output_layer[i].getWeights();

      /* Print Weights */
      for (int j = 0; j < weights.size(); j++) {
         printf(" %4.3f", weights[j]);
      }

      /* Get Bias from Neuron */
      bias = output_layer[i].getBias();

      /* Print Bias */
      printf("   Bias: %4.3f\n", bias);
   }
}

/* Calculate Network Output (Feedforward) */
vector<float> NeuralNetwork::calculate(vector<float> inputs) {
   /* Set Input Nodes */
   setInputs(inputs);

   /* Calculate Hidden Neuron Outputs */
   calculateHidden();

   /* Return Final Neural Network Outputs */
   return calculateOutputs();
}

/* Feed Inputs Into Network */
void NeuralNetwork::setInputs(vector<float> inputs) {
   for (int i = 0; i < input_layer.size(); i++) {
      input_layer[i].setOutput(inputs[i]);
   }
}

/* Calculate Outputs of Hidden Neurons */
void NeuralNetwork::calculateHidden() {
   /* Iterate Through Hidden Layers */
   for (int i = 0; i < hidden_layers.size(); i++) {

      /* Iterate Through Neurons */
      for (int j = 0; j < hidden_layers[i].size(); j++) {

         /* Calculate Output of Neuron */
         hidden_layers[i][j].calculate();
      }
   }
}

/* Calculate Outputs of Network */
vector<float> NeuralNetwork::calculateOutputs() {
   vector<float> output;

   /* Iterate Through Output Neurons */
   for (int i = 0; i < output_layer.size(); i++) {
      /* Calculate Output of Neuron */
      output_layer[i].calculate();
      output.push_back(output_layer[i].getOutput());
   }

   return output;
}

