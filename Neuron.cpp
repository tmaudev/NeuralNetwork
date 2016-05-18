/** Neuron Class
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include "Neuron.h"

Neuron::Neuron() {

}

Neuron::Neuron(vector<Neuron> inputs, int activation_function) {
   input_neurons = inputs;

   for (int i = 0; i < input_neurons.size(); i++) {
      weights.push_back(rand() / (float)RAND_MAX);
   }

   bias = rand() / (float)RAND_MAX;
}

Neuron::Neuron(vector<Neuron> inputs, vector<float> weights, float bias, int activation_function) {
   activation = activation_function;
}

vector<float> Neuron::getWeights() {
   return weights;
}

float Neuron::getBias() {
   return bias;
}

float Neuron::getOutput() {
   return output;
}

void Neuron::setOutput(float value) {
   output = value;
}

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

float Neuron::hyperbolic_tangent(float input, float abs_max, float slope) {
   float output = 0;
   float exp_pos = exp(slope * input);
   float exp_neg = exp(-slope * input);

   output = abs_max * ((exp_pos - exp_neg) / (exp_pos + exp_neg));

   return output;
}