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
      /* Default Neuron Constructor for Input Layer */
      Neuron();

      /* Neuron Constructor with Linear Activation Function */
      Neuron::Neuron(vector<Neuron> inputs)

      /* Neuron Constructor for Non-Input Layer */
      Neuron(vector<Neuron> inputs, int activation_function);

      /* Neuron Constructor to Specify Weights/Biases */
      Neuron(vector<Neuron> inputs, vector<float> weights, float bias, int activation_function);

      /* Return Weights of Neuron */
      vector<float> getWeights();

      /* Return Bias of Neuron */
      float getBias();

      /* Return Last Calculated Output of Neuron */
      float getOutput();

      /* Set Output Value of Neuron (Ex: Input Neurons)*/
      void setOutput(float value);

      /* Calculate Neuron Output */
      void calculate();

   private:
      /* Input Layer */
      vector<Neuron> input_neurons;

      /* Neuron Weights */
      vector<float> weights;

      /* Neuron Bias */
      float bias;

      /* Calculated Output */
      float output;

      /* Activation Function */
      int activation;

      /* Initialize Neuron */
      void initNeuron(vector<Neuron> inputs, int activation_function);

      /* Hyperbolic Tangent Activation Function */
      float hyperbolic_tangent(float input, float abs_max, float slope);
};

#endif