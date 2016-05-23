/** Artificial Neural Network Library Test
  *
  * Author: Tyler Mau
  * Date: May 16, 2016
  */

#include "NeuralNetwork.h"
#define HIDDEN_LAYERS 2

/* Hyperbolic Tangent Activation Function */
float hyperbolic_tangent(float input, float abs_max, float slope) {
   float output = 0;
   float exp_pos = exp(slope * input);
   float exp_neg = exp(-slope * input);

   output = abs_max * ((exp_pos - exp_neg) / (exp_pos + exp_neg));

   return output;
}

int main() {
   vector<int> hidden_layers(HIDDEN_LAYERS);
   vector<float> inputs;
   vector<float> outputs;
   vector< vector<float> > training_input;
   vector< vector<float> > training_output;
   vector<float> *input;
   vector<float> *output;
   int temp;

   /* Create Sample Training Input and Output */
   for (int i = 1; i < 1000; i++) {
      temp = (rand() / (float)RAND_MAX) * 10 + 1;

      input = new vector<float>();
      output = new vector<float>();

      //cout << temp << endl;
      (*input).push_back(temp);
      (*output).push_back(temp * 2);

      training_input.push_back(*input);
      training_output.push_back(*output);
      //cout << temp * temp << endl;
   }

   // input.push_back(5);
   // output.push_back(11);
   // training_input.push_back(input);
   // training_output.push_back(output);

   hidden_layers[0] = 15;
   hidden_layers[1] = 10;

   NeuralNetwork net(1, hidden_layers, 1);

   net.printNetworkSetup();
   //net.printWB();

   //cout << "Test: " << hyperbolic_tangent(0.7, 1, 1) << endl;

   // outputs = net.calculate(input);

   // printf("\nOutput Before Training: %f\n", outputs[0]);

   cout << "Before Training:" << endl;
   for (int i = 0; i < 10; i++) {
      vector<float> test_input, test_output;
      test_input.clear();
      test_input.push_back(i + 1);

      test_output = net.calculate(test_input);

      cout << "Input: " << test_input[0] << "   Output: " << test_output[0] << endl;
   }

   net.train(0.001, 100000, training_input, training_output);

   //outputs = net.calculate(input);

   cout << "\nAfter Training:" << endl;
   for (int i = 0; i < 10; i++) {
      vector<float> test_input, test_output;
      test_input.clear();
      test_input.push_back(i + 1);

      test_output = net.calculate(test_input);

      cout << "Input: " << test_input[0] << "   Output: " << test_output[0] << endl;
   }
   //printf("Output After Training: %f\n\n", outputs[0]);

   //net.printWB();

   return 0;
}