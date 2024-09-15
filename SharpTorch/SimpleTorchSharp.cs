using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using TorchSharp.Data;
using TorchSharp.Utils;
using static TorchSharp.torch.nn;
using System.Diagnostics;
namespace SharpTorch
{
    public class SimpleNN : Module<Tensor, Tensor>
    {
        private Linear fc1; // Input to hidden
        private Linear fc2; // Hidden to output
        private ReLU relu; // Activation function
        private Sigmoid siggy;
        private Softmax softy;

        public SimpleNN() : base("SimpleNN")
        {
            //torch.no_grad();
            // Define layers
            fc1 = Linear(2, 2); // 2 inputs to 2 hidden nodes            
            fc2 = Linear(2, 2); // 2 hidden nodes to 2 outputs
            relu = ReLU();
            siggy = Sigmoid();
            softy = Softmax(2);
            

            // Manually initialize weights
            InitWeights();

            // Register modules
            RegisterComponents();
        }

        public static void DoTrainOnePass()
        {
     
            // Instantiate the model
            var model = new SimpleNN();

            // Define loss function
            var lossFunc = MSELoss(); // Mean Squared Error Loss

            // Example input (batch of size 1, 2 input features)
            var input = torch.tensor(new float[,] { { 0.05f, 0.10f } });

            // Target (2 output values)
            var target = torch.tensor(new float[,] { { 0.01f, 0.99f } });

            // Forward pass
            var output = model.forward(input);
            Debug.WriteLine("\nModel output: " + output.ToString(TorchSharp.TensorStringStyle.Julia));
           
            Debug.WriteLine("Output after the second layer:\n" + output);

            // Compute the loss
            Tensor loss = lossFunc.forward(output, target);
            Debug.WriteLine("\nLoss: " + loss.ToString(TorchSharp.TensorStringStyle.Julia));

            // Backpropagation
            loss.backward();

            // Print gradients after backpropagation
            model.PrintGradientsAndWeights();

            // Manual weight update using gradients
            float learningRate = 0.5f;
            model.ManualWeightUpdate(learningRate);

            // Print updated weights after manual update
            Debug.WriteLine("\nWeights after manual update:");
            model.PrintGradientsAndWeights();
        }

        // Function to manually initialize weights
        // Function to manually initialize weights
        private void InitWeights()
        {
            // Access weights and biases via parameters

            Tensor w1 = torch.from_array(new float[,] {
                { 0.15F, 0.2F },
                { 0.25F, 0.30F }
            });
            Tensor b1 = torch.from_array(new float[] { 0.35f, 0.35f });

            fc1.weight = w1.AsParameter();
            fc1.bias = b1.AsParameter();

            Tensor w2 = torch.from_array(new float[,] {
                { 0.40f, 0.45f },
                { 0.50f, 0.55f }
            });
            Tensor b2 = torch.from_array(new float[] { 0.60f, 0.60f });

            fc2.weight = w2.AsParameter();
            fc2.bias = b2.AsParameter();
        }

        public override Tensor forward(Tensor input)
        {
            // Forward pass through the network
            Tensor x = fc1.forward(input);
            Debug.WriteLine("Output after first layer (before activation):\n" + x.ToString(TorchSharp.TensorStringStyle.Julia));
            x = siggy.forward(x);
            Debug.WriteLine("Output after activation (1st layer):\n" + x.ToString(TorchSharp.TensorStringStyle.Julia));
            x = fc2.forward(x);
            Debug.WriteLine("Output after 2nd layer, pre activation:\n" + x.ToString(TorchSharp.TensorStringStyle.Julia));
            x = siggy.forward(x);
            Debug.WriteLine("Output after 2nd layer activation:\n" + x.ToString(TorchSharp.TensorStringStyle.Julia));
            return x;
        }

        // Helper to print out gradients and weights
        public void PrintGradientsAndWeights()
        {
            Debug.WriteLine("\nGradients after backpropagation:");
            Debug.WriteLine("fc1 weight gradient:\n" + fc1.weight.grad.ToString(TorchSharp.TensorStringStyle.Julia));
            Debug.WriteLine("fc2 weight gradient:\n" + fc2.weight.grad.ToString(TorchSharp.TensorStringStyle.Julia));

            Debug.WriteLine("\nUpdated Weights:");
            Debug.WriteLine("fc1 weights:\n" + fc1.weight.ToString(TorchSharp.TensorStringStyle.Julia));
            Debug.WriteLine("fc2 weights:\n" + fc2.weight.ToString(TorchSharp.TensorStringStyle.Julia));

            Debug.WriteLine("\nBiases after backpropagation:");
            Debug.WriteLine("fc1 bias:\n" + fc1.bias.ToString(TorchSharp.TensorStringStyle.Julia));
            Debug.WriteLine("fc2 bias:\n" + fc2.bias.ToString(TorchSharp.TensorStringStyle.Julia));
        }

        // Function to manually update weights using gradients
        public void ManualWeightUpdate(float learningRate)
        {
            // Subtract a multiple of the gradient from the weights
            using (no_grad())
            {
                Tensor foo = (learningRate * fc1.weight.grad);
                Tensor bar = fc1.weight - foo;
                fc1.weight.set_(bar);

                fc2.weight.set_(fc2.weight - (learningRate * fc2.weight.grad));

                // Update biases as well, if required
                fc1.bias.set_(fc1.bias - learningRate * fc1.bias.grad);
                fc2.bias.set_(fc2.bias - learningRate * fc2.bias.grad);
            }
        }
    }
}

