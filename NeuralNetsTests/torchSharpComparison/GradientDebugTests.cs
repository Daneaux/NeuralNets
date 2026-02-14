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
    public class GradientDebugTests
    {
        private const float Tolerance = 1e-3f;
        private const int RandomSeed = 42;

        [TestMethod]
        public void Debug_GradientComputation_StepByStep()
        {
            // Simple test case: 3 inputs, 2 outputs
            int inputDim = 3;
            int outputDim = 2;
            float learningRate = 0.1f;

            // Simple fixed input and target for reproducibility
            float[] inputData = { 0.5f, 0.3f, 0.2f };
            float[] targetData = { 1.0f, 0.0f };  // One-hot for class 0

            // Simple fixed weights
            float[,] weightsData = new float[,] {
                { 0.1f, 0.2f, 0.3f },  // Neuron 0 weights
                { 0.4f, 0.5f, 0.6f }   // Neuron 1 weights
            };
            float[] biasData = { 0.1f, 0.2f };

            Console.WriteLine("=== Manual Gradient Calculation ===");
            
            // Step 1: Forward pass - compute Z (pre-activation)
            float[] z = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
            {
                z[i] = biasData[i];
                for (int j = 0; j < inputDim; j++)
                {
                    z[i] += weightsData[i, j] * inputData[j];
                }
            }
            Console.WriteLine($"Z (logits): [{z[0]}, {z[1]}]");

            // Step 2: Apply softmax
            float maxZ = z.Max();
            float[] expZ = new float[outputDim];
            float sumExpZ = 0;
            for (int i = 0; i < outputDim; i++)
            {
                expZ[i] = (float)System.Math.Exp(z[i] - maxZ);
                sumExpZ += expZ[i];
            }
            float[] softmax = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
            {
                softmax[i] = expZ[i] / sumExpZ;
            }
            Console.WriteLine($"Softmax: [{softmax[0]}, {softmax[1]}]");

            // Step 3: Compute loss derivative (dL/dZ) = softmax - target
            float[] dL_dZ = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
            {
                dL_dZ[i] = softmax[i] - targetData[i];
            }
            Console.WriteLine($"dL/dZ (loss gradient): [{dL_dZ[0]}, {dL_dZ[1]}]");

            // Step 4: Compute weight gradients (dL/dW = dL/dZ * X^T)
            // This creates an outer product: dL/dW[i,j] = dL/dZ[i] * X[j]
            float[,] manualWeightGrad = new float[outputDim, inputDim];
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    manualWeightGrad[i, j] = dL_dZ[i] * inputData[j];
                }
            }
            Console.WriteLine("\nManual Weight Gradients:");
            for (int i = 0; i < outputDim; i++)
            {
                Console.Write($"  Neuron {i}: [");
                for (int j = 0; j < inputDim; j++)
                {
                    Console.Write($"{manualWeightGrad[i, j]:F6}");
                    if (j < inputDim - 1) Console.Write(", ");
                }
                Console.WriteLine("]");
            }

            // Step 5: Bias gradients are just dL/dZ
            float[] manualBiasGrad = dL_dZ;
            Console.WriteLine($"\nManual Bias Gradients: [{manualBiasGrad[0]:F6}, {manualBiasGrad[1]:F6}]");

            // === TorchSharp ===
            Console.WriteLine("\n=== TorchSharp ===");
            // Clone arrays because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights = torch.from_array((float[,])weightsData.Clone());
            var torchLinear = Linear(inputDim, outputDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torch.from_array((float[])biasData.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, outputDim];
            for (int i = 0; i < outputDim; i++) target2d[0, i] = targetData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);

            var torchPred = torchLinear.forward(torchInput);
            var torchLoss = torchLossFn.forward(torchPred, torchTarget);
            torchLoss.backward();

            var torchGradW = torchLinear.weight.grad.cpu().data<float>().ToArray();
            var torchGradB = torchLinear.bias.grad.cpu().data<float>().ToArray();

            Console.WriteLine("TorchSharp Weight Gradients:");
            for (int i = 0; i < outputDim; i++)
            {
                Console.Write($"  Neuron {i}: [");
                for (int j = 0; j < inputDim; j++)
                {
                    Console.Write($"{torchGradW[i * inputDim + j]:F6}");
                    if (j < inputDim - 1) Console.Write(", ");
                }
                Console.WriteLine("]");
            }
            Console.WriteLine($"TorchSharp Bias Gradients: [{torchGradB[0]:F6}, {torchGradB[1]:F6}]");

            // === NeuralNets ===
            Console.WriteLine("\n=== NeuralNets ===");
            var weights = MatrixFactory.CreateMatrix(weightsData);
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var layer = new WeightedLayer(inputShape, outputDim, weights, biases);

            var layers = new List<Layer> { layer };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(targetData);
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnOutput = layer.FeedFoward(nnInput.ToTensor());
            var nnOutputVec = nnOutput.ToColumnVector();
            Console.WriteLine($"NeuralNets Z (logits): [{layer.Y[0]:F6}, {layer.Y[1]:F6}]");

            // Backward pass
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, nnOutputVec);

            // Get gradients
            var nnWeightGrad = layer.LastWeightGradient;
            var nnBiasGrad = layer.LastBiasGradient;

            Console.WriteLine("NeuralNets Weight Gradients:");
            for (int i = 0; i < outputDim; i++)
            {
                Console.Write($"  Neuron {i}: [");
                for (int j = 0; j < inputDim; j++)
                {
                    Console.Write($"{nnWeightGrad[i, j]:F6}");
                    if (j < inputDim - 1) Console.Write(", ");
                }
                Console.WriteLine("]");
            }
            Console.WriteLine($"NeuralNets Bias Gradients: [{nnBiasGrad[0]:F6}, {nnBiasGrad[1]:F6}]");

            // === Compare ===
            Console.WriteLine("\n=== Comparison ===");
            bool weightsMatch = true;
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    float expected = torchGradW[i * inputDim + j];
                    float actual = nnWeightGrad[i, j];
                    float manual = manualWeightGrad[i, j];
                    float diff = System.Math.Abs(expected - actual);
                    
                    if (diff > Tolerance)
                    {
                        Console.WriteLine($"WEIGHT MISMATCH at [{i},{j}]: Torch={expected:F6}, NeuralNets={actual:F6}, Manual={manual:F6}, Diff={diff:F6}");
                        weightsMatch = false;
                    }
                }
            }

            bool biasesMatch = true;
            for (int i = 0; i < outputDim; i++)
            {
                float diff = System.Math.Abs(torchGradB[i] - nnBiasGrad[i]);
                if (diff > Tolerance)
                {
                    Console.WriteLine($"BIAS MISMATCH at [{i}]: Torch={torchGradB[i]:F6}, NeuralNets={nnBiasGrad[i]:F6}, Manual={manualBiasGrad[i]:F6}, Diff={diff:F6}");
                    biasesMatch = false;
                }
            }

            if (weightsMatch && biasesMatch)
            {
                Console.WriteLine("✓ All gradients match!");
            }
            else
            {
                Console.WriteLine("✗ Gradients do NOT match!");
            }

            // Additional debugging: Check X (input stored in layer)
            Console.WriteLine($"\nDebug - Layer.X (stored input): [{layer.X[0]:F6}, {layer.X[1]:F6}, {layer.X[2]:F6}]");
            Console.WriteLine($"Debug - Original input: [{inputData[0]:F6}, {inputData[1]:F6}, {inputData[2]:F6}]");

            // Assert for test failure/success
            Assert.IsTrue(weightsMatch, "Weight gradients do not match!");
            Assert.IsTrue(biasesMatch, "Bias gradients do not match!");
        }

        [TestMethod]
        public void Debug_OuterProduct_Order()
        {
            // Test to verify outer product order
            // For weight gradient: dL/dW should be outer(dL/dZ, X)
            // Result should be [outputDim x inputDim]
            
            float[] dL_dZ = { 0.5f, -0.3f };  // 2 elements
            float[] X = { 0.1f, 0.2f, 0.3f };  // 3 elements
            
            var dL_dZ_vec = new AvxColumnVector(dL_dZ);
            var X_vec = new AvxColumnVector(X);
            
            // X.RhsOuterProduct(dL_dZ) should give dL_dZ outer X
            var result = X_vec.RhsOuterProduct(dL_dZ_vec.ToTensor());
            
            Console.WriteLine("Outer Product Test:");
            Console.WriteLine($"dL/dZ: [{dL_dZ[0]}, {dL_dZ[1]}]");
            Console.WriteLine($"X: [{X[0]}, {X[1]}, {X[2]}]");
            Console.WriteLine("Result matrix (should be 2x3):");
            Console.WriteLine($"  [{result[0, 0]:F4}, {result[0, 1]:F4}, {result[0, 2]:F4}]");
            Console.WriteLine($"  [{result[1, 0]:F4}, {result[1, 1]:F4}, {result[1, 2]:F4}]");
            
            // Expected:
            // Row 0: 0.5 * [0.1, 0.2, 0.3] = [0.05, 0.10, 0.15]
            // Row 1: -0.3 * [0.1, 0.2, 0.3] = [-0.03, -0.06, -0.09]
            
            Assert.AreEqual(2, result.Rows, "Rows should be outputDim");
            Assert.AreEqual(3, result.Cols, "Cols should be inputDim");
            Assert.AreEqual(0.05f, result[0, 0], 1e-6, "[0,0] should be 0.05");
            Assert.AreEqual(0.10f, result[0, 1], 1e-6, "[0,1] should be 0.10");
            Assert.AreEqual(0.15f, result[0, 2], 1e-6, "[0,2] should be 0.15");
            Assert.AreEqual(-0.03f, result[1, 0], 1e-6, "[1,0] should be -0.03");
            Assert.AreEqual(-0.06f, result[1, 1], 1e-6, "[1,1] should be -0.06");
            Assert.AreEqual(-0.09f, result[1, 2], 1e-6, "[1,2] should be -0.09");
        }
    }
}
