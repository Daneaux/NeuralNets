using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace NeuralNetsTests.torchSharpComparison
{
    [TestClass]
    public class GradientTraceTests
    {
        [TestMethod]
        public void Trace_Layer1_Gradient_Mismatch()
        {
            // Simple 2-layer network: 2 inputs -> 2 hidden -> 2 outputs
            int inputDim = 2;
            int hiddenDim = 2;
            int outputDim = 2;
            float learningRate = 0.1f;

            // Fixed input and target
            float[] inputData = { 0.5f, 0.3f };
            float[] targetData = { 1.0f, 0.0f };  // One-hot for class 0

            // Fixed weights
            float[,] weights1Data = new float[,] { { 0.1f, 0.2f }, { 0.3f, 0.4f } };
            float[] bias1Data = { 0.1f, 0.2f };
            float[,] weights2Data = new float[,] { { 0.15f, 0.25f }, { 0.35f, 0.45f } };
            float[] bias2Data = { 0.05f, 0.15f };

            Console.WriteLine("=== GRADIENT TRACING ===\n");

            // Build NeuralNets network
            var weights1 = MatrixFactory.CreateMatrix(weights1Data);
            var biases1 = new AvxColumnVector(bias1Data);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer1 = new WeightedLayer(inputShape, hiddenDim, weights1, biases1);
            var reluLayer1 = new ReLUActivaction();
            var weights2 = MatrixFactory.CreateMatrix(weights2Data);
            var biases2 = new AvxColumnVector(bias2Data);
            var weightedLayer2 = new WeightedLayer(weightedLayer1.OutputShape, outputDim, weights2, biases2);

            var layers = new System.Collections.Generic.List<Layer> { weightedLayer1, reluLayer1, weightedLayer2 };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            // Forward pass
            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(targetData);
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            Console.WriteLine("FORWARD PASS:");
            var nnZ1 = weightedLayer1.FeedFoward(nnInput.ToTensor());
            Console.WriteLine($"  Z1 (Layer 1 output): [{weightedLayer1.Y[0]:F6}, {weightedLayer1.Y[1]:F6}]");
            
            var nnA1 = reluLayer1.FeedFoward(nnZ1);
            Console.WriteLine($"  A1 (ReLU output): [{reluLayer1.LastActivation.ToColumnVector()[0]:F6}, {reluLayer1.LastActivation.ToColumnVector()[1]:F6}]");
            
            var nnZ2 = weightedLayer2.FeedFoward(nnA1);
            Console.WriteLine($"  Z2 (Layer 2 output): [{weightedLayer2.Y[0]:F6}, {weightedLayer2.Y[1]:F6}]");

            // Compute expected gradients manually
            float[] manualZ1 = { 0.21f, 0.47f };
            float[] manualA1 = { 0.21f, 0.47f };
            float[] manualZ2 = { 0.199f, 0.435f };
            
            // Softmax
            float maxZ2 = manualZ2.Max();
            float[] expZ2 = manualZ2.Select(z => (float)System.Math.Exp(z - maxZ2)).ToArray();
            float sumExp = expZ2.Sum();
            float[] manualSoftmax = expZ2.Select(e => e / sumExp).ToArray();
            
            // Loss gradient (dL/dZ2)
            float[] manualDL_dZ2 = new float[outputDim];
            for (int i = 0; i < outputDim; i++) manualDL_dZ2[i] = manualSoftmax[i] - targetData[i];
            
            Console.WriteLine("\nEXPECTED BACKWARD PASS:");
            Console.WriteLine($"  dL/dZ2 (initial): [{manualDL_dZ2[0]:F6}, {manualDL_dZ2[1]:F6}]");
            
            // Backprop through Layer 2: dL/dA1 = W2^T · dL/dZ2
            float[] manualDL_dA1 = new float[hiddenDim];
            for (int j = 0; j < hiddenDim; j++)
            {
                for (int i = 0; i < outputDim; i++)
                    manualDL_dA1[j] += weights2Data[i, j] * manualDL_dZ2[i];
            }
            Console.WriteLine($"  dL/dA1 (after Layer 2): [{manualDL_dA1[0]:F6}, {manualDL_dA1[1]:F6}]");
            
            // ReLU derivative
            float[] reluDeriv = manualZ1.Select(z => z > 0 ? 1.0f : 0.0f).ToArray();
            Console.WriteLine($"  ReLU derivative mask: [{reluDeriv[0]:F0}, {reluDeriv[1]:F0}]");
            
            // Backprop through ReLU: dL/dZ1 = dL/dA1 ⊙ ReLU'(Z1)
            float[] manualDL_dZ1 = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) manualDL_dZ1[i] = manualDL_dA1[i] * reluDeriv[i];
            Console.WriteLine($"  dL/dZ1 (after ReLU): [{manualDL_dZ1[0]:F6}, {manualDL_dZ1[1]:F6}]");
            
            // Layer 1 weight gradient: dL/dW1 = dL/dZ1 · X^T
            float[,] manualDL_dW1 = new float[hiddenDim, inputDim];
            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < inputDim; j++)
                    manualDL_dW1[i, j] = manualDL_dZ1[i] * inputData[j];
            Console.WriteLine($"  Expected dL/dW1:");
            Console.WriteLine($"    [{manualDL_dW1[0, 0]:F6}, {manualDL_dW1[0, 1]:F6}]");
            Console.WriteLine($"    [{manualDL_dW1[1, 0]:F6}, {manualDL_dW1[1, 1]:F6}]");
            
            // Layer 1 bias gradient: dL/dB1 = dL/dZ1
            Console.WriteLine($"  Expected dL/dB1: [{manualDL_dZ1[0]:F6}, {manualDL_dZ1[1]:F6}]");

            // Now run NeuralNets backward pass
            Console.WriteLine("\nNEURALNETS BACKWARD PASS:");
            var renderContext = new RenderContext(network, 1, null);
            
            // Manually trace what happens in BackProp
            Console.WriteLine("  Computing initial dE/dX (loss derivative)...");
            var lossFn = new CategoricalCrossEntropy();
            var initialGrad = lossFn.Derivative(nnTarget, weightedLayer2.Y);
            Console.WriteLine($"    Initial gradient: [{initialGrad[0]:F6}, {initialGrad[1]:F6}]");
            
            // Call BackProp
            renderContext.BackProp(trainingPair, weightedLayer2.Y);
            
            // Get computed gradients
            var nnGradW1 = weightedLayer1.LastWeightGradient;
            var nnGradB1 = weightedLayer1.LastBiasGradient;
            
            Console.WriteLine("\n  Actual NeuralNets results:");
            Console.WriteLine($"    dL/dW1:");
            Console.WriteLine($"      [{nnGradW1[0, 0]:F6}, {nnGradW1[0, 1]:F6}]");
            Console.WriteLine($"      [{nnGradW1[1, 0]:F6}, {nnGradW1[1, 1]:F6}]");
            Console.WriteLine($"    dL/dB1: [{nnGradB1[0]:F6}, {nnGradB1[1]:F6}]");

            // Compare
            Console.WriteLine("\n=== COMPARISON ===");
            bool allMatch = true;
            
            for (int i = 0; i < hiddenDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    float expected = manualDL_dW1[i, j];
                    float actual = nnGradW1[i, j];
                    float diff = System.Math.Abs(expected - actual);
                    bool match = diff < 1e-3;
                    allMatch &= match;
                    string status = match ? "✓" : "✗";
                    Console.WriteLine($"  dL/dW1[{i},{j}]: Expected={expected:F6}, Actual={actual:F6}, Diff={diff:F6} {status}");
                }
            }
            
            for (int i = 0; i < hiddenDim; i++)
            {
                float expected = manualDL_dZ1[i];
                float actual = nnGradB1[i];
                float diff = System.Math.Abs(expected - actual);
                bool match = diff < 1e-3;
                allMatch &= match;
                string status = match ? "✓" : "✗";
                Console.WriteLine($"  dL/dB1[{i}]: Expected={expected:F6}, Actual={actual:F6}, Diff={diff:F6} {status}");
            }

            if (!allMatch)
            {
                Console.WriteLine("\n✗ Gradient mismatch detected!");
                
                // Calculate ratio to see pattern
                Console.WriteLine("\nRatio analysis (Actual / Expected):");
                for (int i = 0; i < hiddenDim; i++)
                {
                    for (int j = 0; j < inputDim; j++)
                    {
                        float ratio = nnGradW1[i, j] / manualDL_dW1[i, j];
                        Console.WriteLine($"  dL/dW1[{i},{j}]: {ratio:F4}");
                    }
                }
                for (int i = 0; i < hiddenDim; i++)
                {
                    float ratio = nnGradB1[i] / manualDL_dZ1[i];
                    Console.WriteLine($"  dL/dB1[{i}]: {ratio:F4}");
                }
            }

            Assert.IsTrue(allMatch, "Gradient mismatch - see output for details");
        }
    }
}
