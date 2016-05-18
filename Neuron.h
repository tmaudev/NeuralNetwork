/** Neuron Class
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#ifndef NEURON_H

#include <iostream>
#include <vector>
#include <math.h>

#define HT_ACTIVATION 0
#define LINEAR_ACTIVATION 1

#define HYPERBOLIC_TANGENT_MAX 1
#define HYPERBOLIC_TANGENT_SLOPE 1

using namespace std;

class Neuron {
   public:
      Neuron();
      Neuron(vector<Neuron> inputs, int activation_function);
      Neuron(vector<Neuron> inputs, vector<float> weights, float bias, int activation_function);
      vector<float> getWeights();
      float getBias();
      float getOutput();
      void setOutput(float value);
      void calculate();

   private:
      vector<Neuron> input_neurons;
      vector<float> inputs;
      vector<float> weights;
      float bias;
      float output;
      int activation;
      float hyperbolic_tangent(float input, float abs_max, float slope);
};

#endif