using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace NeuralNetsTests.torchSharpComparison
{
    [TestClass]
    public class TwoLayerDebugTests
    {
        private const float Tolerance = 1e-3f;

        [TestMethod]
        public void Debug_TwoLayer_Gradients()
        {
            // Simple 2-layer network: 3 inputs -> 2 hidden -> 2 outputs
            int inputDim = 3;
            int hiddenDim = 2;
            int outputDim = 2;
            float learningRate = 0.1f;

            // Fixed input and target
            float[] inputData = { 0.5f, 0.3f, 0.2f };
            float[] targetData = { 1.0f, 0.0f };  // One-hot for class 0

            // Fixed weights for layer 1 (hidden)
            float[,] weights1Data = new float[,] {
                { 0.1f, 0.2f, 0.3f },  // Neuron 0
                { 0.4f, 0.5f, 0.6f }   // Neuron 1
            };
            float[] bias1Data = { 0.1f, 0.2f };

            // Fixed weights for layer 2 (output)
            float[,] weights2Data = new float[,] {
                { 0.15f, 0.25f },  // Neuron 0
                { 0.35f, 0.45f }   // Neuron 1
            };
            float[] bias2Data = { 0.05f, 0.15f };

            Console.WriteLine("=== Manual Calculation ===");

            // Layer 1 forward
            float[] z1 = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++)
            {
                z1[i] = bias1Data[i];
                for (int j = 0; j < inputDim; j++)
                    z1[i] += weights1Data[i, j] * inputData[j];
            }
            Console.WriteLine($"Z1 (pre-activation): [{z1[0]:F6}, {z1[1]:F6}]");

            // ReLU
            float[] a1 = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++)
                a1[i] = System.Math.Max(0, z1[i]);
            Console.WriteLine($"A1 (ReLU output): [{a1[0]:F6}, {a1[1]:F6}]");

            // ReLU derivative
            float[] reluDeriv = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++)
                reluDeriv[i] = z1[i] > 0 ? 1 : 0;
            Console.WriteLine($"ReLU derivative: [{reluDeriv[0]}, {reluDeriv[1]}]");

            // Layer 2 forward
            float[] z2 = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
            {
                z2[i] = bias2Data[i];
                for (int j = 0; j < hiddenDim; j++)
                    z2[i] += weights2Data[i, j] * a1[j];
            }
            Console.WriteLine($"Z2 (logits): [{z2[0]:F6}, {z2[1]:F6}]");

            // Softmax
            float maxZ2 = z2.Max();
            float[] expZ2 = new float[outputDim];
            float sumExpZ2 = 0;
            for (int i = 0; i < outputDim; i++)
            {
                expZ2[i] = (float)System.Math.Exp(z2[i] - maxZ2);
                sumExpZ2 += expZ2[i];
            }
            float[] softmax = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
                softmax[i] = expZ2[i] / sumExpZ2;
            Console.WriteLine($"Softmax: [{softmax[0]:F6}, {softmax[1]:F6}]");

            // Loss derivative (dL/dZ2) = softmax - target
            float[] dL_dZ2 = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
                dL_dZ2[i] = softmax[i] - targetData[i];
            Console.WriteLine($"dL/dZ2: [{dL_dZ2[0]:F6}, {dL_dZ2[1]:F6}]");

            // Layer 2 gradients
            float[,] dL_dW2 = new float[outputDim, hiddenDim];
            for (int i = 0; i < outputDim; i++)
                for (int j = 0; j < hiddenDim; j++)
                    dL_dW2[i, j] = dL_dZ2[i] * a1[j];
            Console.WriteLine($"\nManual dL/dW2:");
            for (int i = 0; i < outputDim; i++)
            {
                Console.Write($"  [{dL_dW2[i, 0]:F6}, {dL_dW2[i, 1]:F6}]");
                Console.WriteLine();
            }

            float[] dL_dB2 = dL_dZ2;
            Console.WriteLine($"Manual dL/dB2: [{dL_dB2[0]:F6}, {dL_dB2[1]:F6}]");

            // Backprop through layer 2: dL/dA1 = W2^T * dL/dZ2
            float[] dL_dA1 = new float[hiddenDim];
            for (int j = 0; j < hiddenDim; j++)
            {
                dL_dA1[j] = 0;
                for (int i = 0; i < outputDim; i++)
                    dL_dA1[j] += weights2Data[i, j] * dL_dZ2[i];
            }
            Console.WriteLine($"dL/dA1: [{dL_dA1[0]:F6}, {dL_dA1[1]:F6}]");

            // Backprop through ReLU: dL/dZ1 = dL/dA1 * relu_derivative
            float[] dL_dZ1 = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++)
                dL_dZ1[i] = dL_dA1[i] * reluDeriv[i];
            Console.WriteLine($"dL/dZ1: [{dL_dZ1[0]:F6}, {dL_dZ1[1]:F6}]");

            // Layer 1 gradients
            float[,] dL_dW1 = new float[hiddenDim, inputDim];
            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < inputDim; j++)
                    dL_dW1[i, j] = dL_dZ1[i] * inputData[j];
            Console.WriteLine($"\nManual dL/dW1:");
            for (int i = 0; i < hiddenDim; i++)
            {
                Console.Write($"  [{dL_dW1[i, 0]:F6}, {dL_dW1[i, 1]:F6}, {dL_dW1[i, 2]:F6}]");
                Console.WriteLine();
            }

            float[] dL_dB1 = dL_dZ1;
            Console.WriteLine($"Manual dL/dB1: [{dL_dB1[0]:F6}, {dL_dB1[1]:F6}]");

            // === TorchSharp ===
            Console.WriteLine("\n=== TorchSharp ===");
            // Clone arrays because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights1 = torch.from_array((float[,])weights1Data.Clone());
            using var torchWeights2 = torch.from_array((float[,])weights2Data.Clone());
            
            var torchLinear1 = Linear(inputDim, hiddenDim);
            torchLinear1.weight = torchWeights1.AsParameter();
            torchLinear1.bias = torch.from_array((float[])bias1Data.Clone()).AsParameter();
            
            var torchLinear2 = Linear(hiddenDim, outputDim);
            torchLinear2.weight = torchWeights2.AsParameter();
            torchLinear2.bias = torch.from_array((float[])bias2Data.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, outputDim];
            for (int i = 0; i < outputDim; i++) target2d[0, i] = targetData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);

            var torchZ1 = torchLinear1.forward(torchInput);
            using var torchA1 = relu(torchZ1);
            var torchZ2 = torchLinear2.forward(torchA1);
            var torchLoss = torchLossFn.forward(torchZ2, torchTarget);
            torchLoss.backward();

            var torchGradW1 = torchLinear1.weight.grad.cpu().data<float>().ToArray();
            var torchGradB1 = torchLinear1.bias.grad.cpu().data<float>().ToArray();
            var torchGradW2 = torchLinear2.weight.grad.cpu().data<float>().ToArray();
            var torchGradB2 = torchLinear2.bias.grad.cpu().data<float>().ToArray();

            Console.WriteLine("TorchSharp dL/dW2:");
            for (int i = 0; i < outputDim; i++)
            {
                Console.Write($"  [{torchGradW2[i * hiddenDim + 0]:F6}, {torchGradW2[i * hiddenDim + 1]:F6}]");
                Console.WriteLine();
            }
            Console.WriteLine($"TorchSharp dL/dB2: [{torchGradB2[0]:F6}, {torchGradB2[1]:F6}]");

            Console.WriteLine("TorchSharp dL/dW1:");
            for (int i = 0; i < hiddenDim; i++)
            {
                Console.Write($"  [{torchGradW1[i * inputDim + 0]:F6}, {torchGradW1[i * inputDim + 1]:F6}, {torchGradW1[i * inputDim + 2]:F6}]");
                Console.WriteLine();
            }
            Console.WriteLine($"TorchSharp dL/dB1: [{torchGradB1[0]:F6}, {torchGradB1[1]:F6}]");

            // === NeuralNets ===
            Console.WriteLine("\n=== NeuralNets ===");
            var weights1 = MatrixFactory.CreateMatrix(weights1Data);
            var biases1 = new AvxColumnVector(bias1Data);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer1 = new WeightedLayer(inputShape, hiddenDim, weights1, biases1);
            var reluLayer1 = new ReLUActivaction();

            var weights2 = MatrixFactory.CreateMatrix(weights2Data);
            var biases2 = new AvxColumnVector(bias2Data);
            var weightedLayer2 = new WeightedLayer(weightedLayer1.OutputShape, outputDim, weights2, biases2);

            var layers = new List<Layer> { weightedLayer1, reluLayer1, weightedLayer2 };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(targetData);
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnZ1 = weightedLayer1.FeedFoward(nnInput.ToTensor());
            Console.WriteLine($"NeuralNets Z1: [{weightedLayer1.Y[0]:F6}, {weightedLayer1.Y[1]:F6}]");
            
            var nnA1 = reluLayer1.FeedFoward(nnZ1);
            Console.WriteLine($"NeuralNets A1: [{reluLayer1.LastActivation.ToColumnVector()[0]:F6}, {reluLayer1.LastActivation.ToColumnVector()[1]:F6}]");
            
            var nnZ2 = weightedLayer2.FeedFoward(nnA1);
            Console.WriteLine($"NeuralNets Z2: [{weightedLayer2.Y[0]:F6}, {weightedLayer2.Y[1]:F6}]");

            // Backward pass
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, weightedLayer2.Y);

            Console.WriteLine("NeuralNets dL/dW2:");
            for (int i = 0; i < outputDim; i++)
            {
                Console.Write($"  [{weightedLayer2.LastWeightGradient[i, 0]:F6}, {weightedLayer2.LastWeightGradient[i, 1]:F6}]");
                Console.WriteLine();
            }
            Console.WriteLine($"NeuralNets dL/dB2: [{weightedLayer2.LastBiasGradient[0]:F6}, {weightedLayer2.LastBiasGradient[1]:F6}]");

            Console.WriteLine("NeuralNets dL/dW1:");
            for (int i = 0; i < hiddenDim; i++)
            {
                Console.Write($"  [{weightedLayer1.LastWeightGradient[i, 0]:F6}, {weightedLayer1.LastWeightGradient[i, 1]:F6}, {weightedLayer1.LastWeightGradient[i, 2]:F6}]");
                Console.WriteLine();
            }
            Console.WriteLine($"NeuralNets dL/dB1: [{weightedLayer1.LastBiasGradient[0]:F6}, {weightedLayer1.LastBiasGradient[1]:F6}]");

            // === Comparison ===
            Console.WriteLine("\n=== Comparison ===");
            bool allMatch = true;

            // Compare W2
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < hiddenDim; j++)
                {
                    float expected = torchGradW2[i * hiddenDim + j];
                    float actual = weightedLayer2.LastWeightGradient[i, j];
                    float diff = System.Math.Abs(expected - actual);
                    if (diff > Tolerance)
                    {
                        Console.WriteLine($"W2[{i},{j}] MISMATCH: Torch={expected:F6}, NeuralNets={actual:F6}, Diff={diff:F6}");
                        allMatch = false;
                    }
                }
            }

            // Compare B2
            for (int i = 0; i < outputDim; i++)
            {
                float diff = System.Math.Abs(torchGradB2[i] - weightedLayer2.LastBiasGradient[i]);
                if (diff > Tolerance)
                {
                    Console.WriteLine($"B2[{i}] MISMATCH: Torch={torchGradB2[i]:F6}, NeuralNets={weightedLayer2.LastBiasGradient[i]:F6}, Diff={diff:F6}");
                    allMatch = false;
                }
            }

            // Compare W1
            for (int i = 0; i < hiddenDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    float expected = torchGradW1[i * inputDim + j];
                    float actual = weightedLayer1.LastWeightGradient[i, j];
                    float diff = System.Math.Abs(expected - actual);
                    if (diff > Tolerance)
                    {
                        Console.WriteLine($"W1[{i},{j}] MISMATCH: Torch={expected:F6}, NeuralNets={actual:F6}, Diff={diff:F6}");
                        allMatch = false;
                    }
                }
            }

            // Compare B1
            for (int i = 0; i < hiddenDim; i++)
            {
                float diff = System.Math.Abs(torchGradB1[i] - weightedLayer1.LastBiasGradient[i]);
                if (diff > Tolerance)
                {
                    Console.WriteLine($"B1[{i}] MISMATCH: Torch={torchGradB1[i]:F6}, NeuralNets={weightedLayer1.LastBiasGradient[i]:F6}, Diff={diff:F6}");
                    allMatch = false;
                }
            }

            if (allMatch)
                Console.WriteLine("✓ All gradients match!");
            else
                Console.WriteLine("✗ Some gradients do not match!");

            Assert.IsTrue(allMatch, "Gradients do not match - see output for details");
        }
    }
}
