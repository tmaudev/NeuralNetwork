/** Artificial Neural Network Library
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include "NeuralNetwork.h"

/* Neural Network Constructor */
NeuralNetwork::NeuralNetwork(int inputs, vector<int> hidden_config, int outputs) {
   /* Temporary Vector for Creating Layers */
   vector<Neuron> *layer;

   //THROW ERRORS IF WRONG INPUTS

   /* Save Network Configuration */
   num_hidden_layers = hidden_config.size();
   num_inputs = inputs;
   num_outputs = outputs;

   /* Create Vector of Input Nodes */
   layer = new vector<Neuron>();
   for (int i = 0; i < num_inputs; i++) {
      (*layer).push_back(Neuron());
   }
   input_layer = layer;

   /* Create Vector of Hidden Layers */
   for (int i = 0; i < num_hidden_layers; i++) {
      layer = new vector<Neuron>();
      
      /* Create Vector of Nodes for Each Layer */
      for (int j = 0; j < hidden_config[i]; j++) {
         /* Previous Layer is a Hidden Layer */
         if (i) {
            (*layer).push_back(Neuron(hidden_layers[i - 1], HT_ACTIVATION));
         }
         /* Previous Layer is Input Layer */
         else {
            (*layer).push_back(Neuron(input_layer, HT_ACTIVATION));
         }
      }
      hidden_layers.push_back(layer);
      layer_cost.push_back(0);
   }

   /* Create Output Layer (Linear Activation Function) */
   layer = new vector<Neuron>();
   for (int i = 0; i < outputs; i++) {
      (*layer).push_back(Neuron(hidden_layers[num_hidden_layers - 1], LINEAR_ACTIVATION));
   }
   hidden_layers.push_back(layer);
   layer_cost.push_back(0);
   output_layer = layer;
}

/* Print Number of Neurons in Each Layer */
void NeuralNetwork::printNetworkSetup() {
   /* Print Size of Input Layer */
   cout << "Layer Configuration: " << num_inputs;

   /* Print Size of Each Hidden Layer */
   for (int i = 0; i < num_hidden_layers; i++) {
      cout << " " << (*hidden_layers[i]).size();
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
      for (int j = 0; j < (*hidden_layers[i]).size(); j++) {
         cout << "   Neuron " << j << ":";

         /* Get Weights from Neuron */
         weights = (*hidden_layers[i])[j].getWeights();

         /* Print Weights */
         for (int k = 0; k < weights.size(); k++) {
            printf(" %5.5f", weights[k]);
         }

         /* Get Bias from Neuron */
         bias = (*hidden_layers[i])[j].getBias();

         /* Print Bias */
         printf("   Bias: %5.5f\n", bias);
      }
      cout << endl;
   }

   cout << "Output Layer:" << endl;

   /* Iterate Through Output Layer */
   for (int i = 0; i < num_outputs; i++) {
      cout << "   Neuron " << i << ":";

      /* Get Weights from Neuron */
      weights = (*output_layer)[i].getWeights();

      /* Print Weights */
      for (int j = 0; j < weights.size(); j++) {
         printf(" %5.5f", weights[j]);
      }

      /* Get Bias from Neuron */
      bias = (*output_layer)[i].getBias();

      /* Print Bias */
      printf("   Bias: %5.5f\n", bias);
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
   for (int i = 0; i < (*input_layer).size(); i++) {
      (*input_layer)[i].setOutput(inputs[i]);
   }
}

/* Calculate Outputs of Hidden Neurons */
void NeuralNetwork::calculateHidden() {
   /* Iterate Through Hidden Layers */
   for (int i = 0; i < hidden_layers.size(); i++) {

      /* Iterate Through Neurons */
      for (int j = 0; j < (*hidden_layers[i]).size(); j++) {
         /* Calculate Output of Neuron */
         (*hidden_layers[i])[j].calculate();
      }
   }
}

/* Calculate Outputs of Network */
vector<float> NeuralNetwork::calculateOutputs() {
   vector<float> output;

   /* Iterate Through Output Neurons */
   for (int i = 0; i < (*output_layer).size(); i++) {
      /* Calculate Output of Neuron */
      (*output_layer)[i].calculate();
      output.push_back((*output_layer)[i].getOutput());
   }

   return output;
}



void NeuralNetwork::updateHiddenLayers() {
   //for (int i = ) {

   //}
}

// void NeuralNetwork::updateOutputLayer(vector<float> training_output) {
//    float error, deriv_act, gradient, actual_output;
//    vector<float> weights;

//    /* Iterate Through Each Neuron */
//    for (int i = 0; i < (*output_layer).size(); i++) {
//       actual_output = (*output_layer)[i].getOutput();
//       error = training_output[i] - actual_output;
//       deriv_act = hyperbolicTangentDerivative(actual_output);

//       /* Iterate Through Each Weight */
//       for (int j = 0; j < (*output_layer)[i].getWeights().size(); j++) {
//          gradient = -error * deriv_act * actual_output;
//          (*output_layer)[i].updateWeight(j, -training_step * gradient);
//       }
//    }
// }

// void NeuralNetwork::updateWeight(int layer, ) {

// }

void NeuralNetwork::prepareUpdate(vector<float> training_output) {
   float error, phi_deriv;
   vector<float> weights;

   /* Calculate Activation Function Derivatives */
   for (int i = 0; i <= num_hidden_layers; i++) {
      for (int j = 0; j < (*hidden_layers[i]).size(); j++) {
         (*hidden_layers[i])[j].calculatePhiDeriv();
      }
   }

   /* Calculate Layer Costs */
   for (int i = num_hidden_layers; i >= 0; i--) {
      layer_cost[i] = 0;
      for (int j = 0; j < (*hidden_layers[i]).size(); j++) {
         weights = (*hidden_layers[i])[j].getWeights();
         /* Output Layer */
         if (i == num_hidden_layers) {
            error = training_output[j] - (*hidden_layers[i])[j].getOutput();
            phi_deriv = (*hidden_layers[i])[j].getPhiDeriv();
            layer_cost[i] += -error * phi_deriv * weights[j];
         }
         else {

         }
      }
   }   
}

/* Train Neural Network via Supervised Training */
void NeuralNetwork::train(float step, int epoch, vector< vector<float> > training_input, vector< vector<float> > training_output) {
   training_step = step;

   for (int i = 0; i < epoch; i++) {
      //RANDOMIZE ORDER??

      //Calculate Network
      calculate(training_input[i % epoch]);

      prepareUpdate(training_output[i % epoch]);

      //Update Ouput Layer Node Weights
      //updateOutputLayer(training_output[i % epoch]);

      //Update Hidden Layers
      updateHiddenLayers();
   }
}