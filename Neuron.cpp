/** Neuron Class
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include "Neuron.h"

/* Default Neuron Constructor for Input Layer */
Neuron::Neuron() {
}

/* Neuron Constructor with Linear Activation Function */
Neuron::Neuron(vector<Neuron> inputs) {
   initNeuron(inputs, LINEAR_ACTIVATION);
}

/* Neuron Constructor for Non-Input Layer */
Neuron::Neuron(vector<Neuron> inputs, int activation_function) {
   initNeuron(inputs, activation_function);
}

/* Neuron Constructor to Specify Weights/Biases */
/*Neuron::Neuron(vector<Neuron> inputs, vector<float> weights, float bias, int activation_function) {
   activation = activation_function;
}*/

/* Initialize Neuron */
void Neuron::initNeuron(vector<Neuron> inputs, int activation_function) {
  input_neurons = inputs;

   for (int i = 0; i < input_neurons.size(); i++) {
      weights.push_back(rand() / (float)RAND_MAX);
   }

   bias = rand() / (float)RAND_MAX;

   activation = activation_function;

   output = 0;
}


/* Return Weights of Neuron */
vector<float> Neuron::getWeights() {
   return weights;
}

/* Return Bias of Neuron */
float Neuron::getBias() {
   return bias;
}

/* Return Last Calculated Output of Neuron */
float Neuron::getOutput() {
   return output;
}

/* Set Output Value of Neuron (Ex: Input Neurons)*/
void Neuron::setOutput(float value) {
   output = value;
}

/* Calculate Neuron Output */
void Neuron::calculate() {
   float sum = 0;

   for (int i = 0; i < input_neurons.size(); i++) {
      sum += input_neurons[i].getOutput() * weights[i];
   }

   sum += bias;

   if (activation == HT_ACTIVATION) {
      output = hyperbolic_tangent(sum, HYPERBOLIC_TANGENT_MAX, HYPERBOLIC_TANGENT_SLOPE);
   }
   
   output = sum;
}

/* Hyperbolic Tangent Activation Function */
float Neuron::hyperbolic_tangent(float input, float abs_max, float slope) {
   float output = 0;
   float exp_pos = exp(slope * input);
   float exp_neg = exp(-slope * input);

   output = abs_max * ((exp_pos - exp_neg) / (exp_pos + exp_neg));

   return output;
}