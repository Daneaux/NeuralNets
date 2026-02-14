using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NeuralNetsTests.torchSharpComparison
{
    [TestClass]
    public class CrossEntropyLossTests
    {
        private const float Tolerance = 1e-4f;

        /// <summary>
        /// Manual softmax computation for verification
        /// </summary>
        private static float[] ManualSoftmax(float[] logits)
        {
            float max = logits.Max();
            float[] expValues = new float[logits.Length];
            float sumExp = 0;
            
            for (int i = 0; i < logits.Length; i++)
            {
                expValues[i] = (float)System.Math.Exp(logits[i] - max);
                sumExp += expValues[i];
            }
            
            float[] softmax = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                softmax[i] = expValues[i] / sumExp;
            }
            
            return softmax;
        }

        /// <summary>
        /// Manual cross-entropy loss computation
        /// Loss = -sum(target_i * log(softmax_i))
        /// </summary>
        private static float ManualCrossEntropyLoss(float[] target, float[] logits)
        {
            float[] softmax = ManualSoftmax(logits);
            float loss = 0;
            
            for (int i = 0; i < target.Length; i++)
            {
                // Add small epsilon to avoid log(0)
                float epsilon = 1e-7f;
                loss -= target[i] * (float)System.Math.Log(softmax[i] + epsilon);
            }
            
            return loss;
        }

        /// <summary>
        /// Manual gradient computation for cross-entropy with softmax
        /// dL/dz_i = softmax_i - target_i
        /// </summary>
        private static float[] ManualCrossEntropyGradient(float[] target, float[] logits)
        {
            float[] softmax = ManualSoftmax(logits);
            float[] gradient = new float[target.Length];
            
            for (int i = 0; i < target.Length; i++)
            {
                gradient[i] = softmax[i] - target[i];
            }
            
            return gradient;
        }

        [TestMethod]
        public void CrossEntropyLoss_FeedForward_SimpleCase()
        {
            // Simple test case
            float[] logits = { 1.0f, 2.0f, 3.0f };
            float[] target = { 0.0f, 1.0f, 0.0f };  // One-hot for class 1

            // Manual computation
            float manualLoss = ManualCrossEntropyLoss(target, logits);
            float[] manualSoftmax = ManualSoftmax(logits);

            Console.WriteLine("=== CrossEntropyLoss FeedForward Test ===");
            Console.WriteLine($"Logits: [{string.Join(", ", logits)}]");
            Console.WriteLine($"Target: [{string.Join(", ", target)}]");
            Console.WriteLine($"\nManual Softmax: [{string.Join(", ", manualSoftmax.Select(x => x.ToString("F6")))}]");
            Console.WriteLine($"Manual Loss: {manualLoss:F6}");

            // TorchSharp computation
            using var torchLogits = torch.from_array(new float[,] { { logits[0], logits[1], logits[2] } });
            using var torchTarget = torch.from_array(new long[] { 1 });  // Class index
            using var lossFn = CrossEntropyLoss(reduction: Reduction.Sum);
            
            var torchLoss = lossFn.forward(torchLogits, torchTarget);
            float torchLossValue = torchLoss.item<float>();
            
            Console.WriteLine($"\nTorchSharp Loss: {torchLossValue:F6}");

            // NeuralNets computation
            var nnLogits = new AvxColumnVector(logits);
            var nnTarget = new AvxColumnVector(target);
            var lossFunction = new CategoricalCrossEntropy();
            
            float nnLoss = lossFunction.ScalarLoss(nnTarget, nnLogits);
            
            Console.WriteLine($"NeuralNets Loss: {nnLoss:F6}");

            // Compare losses
            Console.WriteLine($"\nLoss Comparison:");
            Console.WriteLine($"  Manual vs TorchSharp: {System.Math.Abs(manualLoss - torchLossValue):F6}");
            Console.WriteLine($"  Manual vs NeuralNets: {System.Math.Abs(manualLoss - nnLoss):F6}");
            Console.WriteLine($"  TorchSharp vs NeuralNets: {System.Math.Abs(torchLossValue - nnLoss):F6}");

            Assert.AreEqual(manualLoss, torchLossValue, Tolerance, 
                $"Manual loss ({manualLoss:F6}) doesn't match TorchSharp ({torchLossValue:F6})");
            Assert.AreEqual(manualLoss, nnLoss, Tolerance, 
                $"Manual loss ({manualLoss:F6}) doesn't match NeuralNets ({nnLoss:F6})");
            Assert.AreEqual(torchLossValue, nnLoss, Tolerance, 
                $"TorchSharp loss ({torchLossValue:F6}) doesn't match NeuralNets ({nnLoss:F6})");

            Console.WriteLine("\n✓ All loss values match!");
        }

        [TestMethod]
        public void CrossEntropyLoss_Gradient_SimpleCase()
        {
            // Simple test case
            float[] logits = { 1.0f, 2.0f, 3.0f };
            float[] target = { 0.0f, 1.0f, 0.0f };  // One-hot for class 1

            // Manual gradient computation
            float[] manualGradient = ManualCrossEntropyGradient(target, logits);

            Console.WriteLine("\n=== CrossEntropyLoss Gradient Test ===");
            Console.WriteLine($"Logits: [{string.Join(", ", logits)}]");
            Console.WriteLine($"Target: [{string.Join(", ", target)}]");
            Console.WriteLine($"\nManual Gradient: [{string.Join(", ", manualGradient.Select(x => x.ToString("F6")))}]");

            // TorchSharp gradient computation
            using var torchLogits = torch.from_array(new float[,] { { logits[0], logits[1], logits[2] } });
            using var torchTarget = torch.from_array(new long[] { 1 });  // Class index
            using var lossFn = CrossEntropyLoss(reduction: Reduction.Sum);
            
            torchLogits.requires_grad = true;
            var torchLoss = lossFn.forward(torchLogits, torchTarget);
            torchLoss.backward();
            
            var torchGradient = torchLogits.grad.cpu().data<float>().ToArray();
            
            Console.WriteLine($"TorchSharp Gradient: [{string.Join(", ", torchGradient.Select(x => x.ToString("F6")))}]");

            // NeuralNets gradient computation
            var nnLogits = new AvxColumnVector(logits);
            var nnTarget = new AvxColumnVector(target);
            var lossFunction = new CategoricalCrossEntropy();
            
            var nnGradient = lossFunction.Derivative(nnTarget, nnLogits);
            float[] nnGradientArray = new float[nnGradient.Size];
            for (int i = 0; i < nnGradient.Size; i++)
            {
                nnGradientArray[i] = nnGradient[i];
            }
            
            Console.WriteLine($"NeuralNets Gradient: [{string.Join(", ", nnGradientArray.Select(x => x.ToString("F6")))}]");

            // Compare gradients
            Console.WriteLine($"\nGradient Comparison:");
            bool allMatch = true;
            for (int i = 0; i < logits.Length; i++)
            {
                float diffManualTorch = System.Math.Abs(manualGradient[i] - torchGradient[i]);
                float diffManualNN = System.Math.Abs(manualGradient[i] - nnGradientArray[i]);
                float diffTorchNN = System.Math.Abs(torchGradient[i] - nnGradientArray[i]);
                
                Console.WriteLine($"  Index {i}:");
                Console.WriteLine($"    Manual={manualGradient[i]:F6}, Torch={torchGradient[i]:F6}, NN={nnGradientArray[i]:F6}");
                Console.WriteLine($"    Diff(Manual,Torch)={diffManualTorch:F6}, Diff(Manual,NN)={diffManualNN:F6}, Diff(Torch,NN)={diffTorchNN:F6}");
                
                if (diffManualTorch > Tolerance || diffManualNN > Tolerance || diffTorchNN > Tolerance)
                {
                    allMatch = false;
                }
            }

            // Assert gradients match
            for (int i = 0; i < logits.Length; i++)
            {
                Assert.AreEqual(manualGradient[i], torchGradient[i], Tolerance, 
                    $"Manual gradient[{i}] ({manualGradient[i]:F6}) doesn't match TorchSharp ({torchGradient[i]:F6})");
                Assert.AreEqual(manualGradient[i], nnGradientArray[i], Tolerance, 
                    $"Manual gradient[{i}] ({manualGradient[i]:F6}) doesn't match NeuralNets ({nnGradientArray[i]:F6})");
                Assert.AreEqual(torchGradient[i], nnGradientArray[i], Tolerance, 
                    $"TorchSharp gradient[{i}] ({torchGradient[i]:F6}) doesn't match NeuralNets ({nnGradientArray[i]:F6})");
            }

            Console.WriteLine("\n✓ All gradient values match!");
        }

        [TestMethod]
        public void CrossEntropyLoss_EndToEnd_SingleLayer()
        {
            // End-to-end test with a single linear layer
            int inputDim = 4;
            int outputDim = 3;
            float learningRate = 0.1f;

            // Fixed input and target
            float[] inputData = { 0.5f, 0.3f, 0.2f, 0.1f };
            float[] targetData = { 0.0f, 1.0f, 0.0f };  // One-hot for class 1

            // Fixed weights
            float[,] weightsData = new float[,] {
                { 0.1f, 0.2f, 0.3f, 0.4f },
                { 0.5f, 0.6f, 0.7f, 0.8f },
                { 0.9f, 1.0f, 1.1f, 1.2f }
            };
            float[] biasData = { 0.1f, 0.2f, 0.3f };

            Console.WriteLine("\n=== End-to-End Single Layer Test ===");

            // Manual forward pass
            float[] manualLogits = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
            {
                manualLogits[i] = biasData[i];
                for (int j = 0; j < inputDim; j++)
                {
                    manualLogits[i] += weightsData[i, j] * inputData[j];
                }
            }
            Console.WriteLine($"Manual Logits: [{string.Join(", ", manualLogits.Select(x => x.ToString("F6")))}]");

            float manualLoss = ManualCrossEntropyLoss(targetData, manualLogits);
            float[] manualGradient = ManualCrossEntropyGradient(targetData, manualLogits);
            
            Console.WriteLine($"Manual Loss: {manualLoss:F6}");
            Console.WriteLine($"Manual Gradient: [{string.Join(", ", manualGradient.Select(x => x.ToString("F6")))}]");

            // TorchSharp forward and backward
            // Clone arrays because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights = torch.from_array((float[,])weightsData.Clone());
            var torchLinear = Linear(inputDim, outputDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torch.from_array((float[])biasData.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(new long[] { 1 });  // Class 1
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);

            var torchLogits = torchLinear.forward(torchInput);
            var torchLoss = torchLossFn.forward(torchLogits, torchTarget);
            torchLoss.backward();

            float torchLossValue = torchLoss.item<float>();
            var torchLogitsData = torchLogits.cpu().data<float>().ToArray();
            var torchGradW = torchLinear.weight.grad.cpu().data<float>().ToArray();
            var torchGradB = torchLinear.bias.grad.cpu().data<float>().ToArray();

            Console.WriteLine($"\nTorchSharp Logits: [{string.Join(", ", torchLogitsData.Select(x => x.ToString("F6")))}]");
            Console.WriteLine($"TorchSharp Loss: {torchLossValue:F6}");

            // NeuralNets forward and backward
            var weights = MatrixFactory.CreateMatrix(weightsData);
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var layer = new WeightedLayer(inputShape, outputDim, weights, biases);

            var layers = new System.Collections.Generic.List<Layer> { layer };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(targetData);
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnOutput = layer.FeedFoward(nnInput.ToTensor());
            var nnLogits = layer.Y;
            
            Console.WriteLine($"\nNeuralNets Logits: [{string.Join(", ", Enumerable.Range(0, outputDim).Select(i => nnLogits[i].ToString("F6")))}]");

            // Compute loss
            var lossFunction = new CategoricalCrossEntropy();
            float nnLoss = lossFunction.ScalarLoss(nnTarget, nnLogits);
            Console.WriteLine($"NeuralNets Loss: {nnLoss:F6}");

            // Backward pass
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, nnLogits);

            // Get gradients
            var nnWeightGrad = layer.LastWeightGradient;
            var nnBiasGrad = layer.LastBiasGradient;

            // Compare logits
            Console.WriteLine($"\n=== Comparison ===");
            for (int i = 0; i < outputDim; i++)
            {
                Assert.AreEqual(manualLogits[i], torchLogitsData[i], Tolerance, 
                    $"Logits[{i}] mismatch between Manual and TorchSharp");
                Assert.AreEqual(manualLogits[i], nnLogits[i], Tolerance, 
                    $"Logits[{i}] mismatch between Manual and NeuralNets");
            }
            Console.WriteLine("✓ Logits match!");

            // Compare losses
            Assert.AreEqual(manualLoss, torchLossValue, Tolerance, 
                "Loss mismatch between Manual and TorchSharp");
            Assert.AreEqual(manualLoss, nnLoss, Tolerance, 
                "Loss mismatch between Manual and NeuralNets");
            Console.WriteLine("✓ Loss values match!");

            // Compare weight gradients
            Console.WriteLine("\nWeight Gradients:");
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    float expected = torchGradW[i * inputDim + j];
                    float actual = nnWeightGrad[i, j];
                    float diff = System.Math.Abs(expected - actual);
                    
                    Console.WriteLine($"  W[{i},{j}]: Torch={expected:F6}, NN={actual:F6}, Diff={diff:F6}");
                    
                    Assert.AreEqual(expected, actual, Tolerance, 
                        $"Weight gradient[{i},{j}] mismatch: Torch={expected:F6}, NN={actual:F6}");
                }
            }
            Console.WriteLine("✓ Weight gradients match!");

            // Compare bias gradients
            Console.WriteLine("\nBias Gradients:");
            for (int i = 0; i < outputDim; i++)
            {
                float expected = torchGradB[i];
                float actual = nnBiasGrad[i];
                float diff = System.Math.Abs(expected - actual);
                
                Console.WriteLine($"  B[{i}]: Torch={expected:F6}, NN={actual:F6}, Diff={diff:F6}");
                
                Assert.AreEqual(expected, actual, Tolerance, 
                    $"Bias gradient[{i}] mismatch: Torch={expected:F6}, NN={actual:F6}");
            }
            Console.WriteLine("✓ Bias gradients match!");

            Console.WriteLine("\n✓ All end-to-end tests passed!");
        }

        [TestMethod]
        public void CrossEntropyLoss_ExtremeValues()
        {
            // Test with extreme logits (very positive and very negative)
            float[] logits = { -10.0f, 0.0f, 10.0f };
            float[] target = { 0.0f, 0.0f, 1.0f };  // One-hot for class 2

            Console.WriteLine("\n=== Extreme Values Test ===");
            Console.WriteLine($"Logits: [{string.Join(", ", logits)}]");
            Console.WriteLine($"Target: [{string.Join(", ", target)}]");

            // Manual computation
            float[] manualSoftmax = ManualSoftmax(logits);
            float manualLoss = ManualCrossEntropyLoss(target, logits);
            float[] manualGradient = ManualCrossEntropyGradient(target, logits);

            Console.WriteLine($"\nManual Softmax: [{string.Join(", ", manualSoftmax.Select(x => x.ToString("F6")))}]");
            Console.WriteLine($"Manual Loss: {manualLoss:F6}");
            Console.WriteLine($"Manual Gradient: [{string.Join(", ", manualGradient.Select(x => x.ToString("F6")))}]");

            // TorchSharp computation
            using var torchLogits = torch.from_array(new float[,] { { logits[0], logits[1], logits[2] } });
            using var torchTarget = torch.from_array(new long[] { 2 });  // Class index 2
            using var lossFn = CrossEntropyLoss(reduction: Reduction.Sum);
            
            var torchLoss = lossFn.forward(torchLogits, torchTarget);
            float torchLossValue = torchLoss.item<float>();
            
            Console.WriteLine($"\nTorchSharp Loss: {torchLossValue:F6}");

            // NeuralNets computation
            var nnLogits = new AvxColumnVector(logits);
            var nnTarget = new AvxColumnVector(target);
            var lossFunction = new CategoricalCrossEntropy();
            
            float nnLoss = lossFunction.ScalarLoss(nnTarget, nnLogits);
            var nnGradient = lossFunction.Derivative(nnTarget, nnLogits);
            float[] nnGradientArray = new float[nnGradient.Size];
            for (int i = 0; i < nnGradient.Size; i++)
            {
                nnGradientArray[i] = nnGradient[i];
            }
            
            Console.WriteLine($"NeuralNets Loss: {nnLoss:F6}");
            Console.WriteLine($"NeuralNets Gradient: [{string.Join(", ", nnGradientArray.Select(x => x.ToString("F6")))}]");

            // Verify numerical stability - softmax should sum to 1
            float softmaxSum = manualSoftmax.Sum();
            Console.WriteLine($"\nSoftmax sum: {softmaxSum:F6} (should be ~1.0)");
            Assert.AreEqual(1.0f, softmaxSum, 1e-5f, "Softmax values don't sum to 1");

            // Verify extreme values handled correctly
            Assert.AreEqual(manualLoss, torchLossValue, Tolerance * 10, 
                "Loss mismatch with extreme values");
            Assert.AreEqual(manualLoss, nnLoss, Tolerance * 10, 
                "NeuralNets loss mismatch with extreme values");

            Console.WriteLine("\n✓ Extreme values test passed!");
        }
    }
}
