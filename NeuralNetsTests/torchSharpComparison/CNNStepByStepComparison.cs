using MatrixLibrary;
using MnistReader_ANN;
using NeuralNets;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NeuralNetsTests.torchSharpComparison
{
    /// <summary>
    /// Detailed step-by-step CNN comparison to find divergence points
    /// </summary>
    [TestClass]
    public class CNNStepByStepComparison
    {
        private const float Tolerance = 1e-4f;

        /// <summary>
        /// Compares a single training step in detail to find where gradients/weights diverge
        /// </summary>
        [TestMethod]
        public void CNN_SingleStep_DetailedComparison()
        {
            Console.WriteLine("\n=== CNN Single Step Detailed Comparison ===\n");

            // Get single MNIST sample
            var trainingSet = new MNISTTrainingSet();
            var pair = trainingSet.BuildNewRandomizedTrainingList(do2DImage: true).First();
            var inputData = pair.Input.Matrices[0].Mat;
            int trueLabel = ArgMax(pair.Output.ToColumnVector()!.Column);

            Console.WriteLine($"Sample label: {trueLabel}");
            Console.WriteLine($"Input shape: 28x28\n");

            // Initialize both networks with SAME weights (all zeros for simplicity)
            float learningRate = 0.001f;

            // ==================== TORCHSHARP ====================
            Console.WriteLine("--- TorchSharp Setup ---");
            var torchConv = Conv2d((long)1, (long)5, (long)4, stride: (long)1, padding: (long)0);
            var torchPool = torch.nn.MaxPool2d((long)2, stride: (long)2);
            var torchDense = Linear(720, 10);

            // Zero initialization
            torchConv.weight = torch.zeros(5, 1, 4, 4).AsParameter();
            torchConv.bias = torch.zeros(5).AsParameter();
            torchDense.weight = torch.zeros(10, 720).AsParameter();
            torchDense.bias = torch.zeros(10).AsParameter();

            // Forward
            using var torchInput = torch.from_array(inputData).reshape(1, 1, 28, 28);
            var tConv = torchConv.forward(torchInput);
            var tRelu = relu(tConv);
            var tPool = torchPool.forward(tRelu);
            var tFlat = tPool.reshape(1, 720);
            var tLogits = torchDense.forward(tFlat);

            Console.WriteLine($"Conv output shape: {string.Join("x", tConv.shape)}");
            Console.WriteLine($"Pool output shape: {string.Join("x", tPool.shape)}");
            Console.WriteLine($"Logits (first 5): [{string.Join(", ", tLogits.cpu().data<float>().ToArray().Take(5).Select(x => x.ToString("F6")))}...]");

            // Loss and backward
            using var torchTarget = torch.from_array(new long[] { trueLabel });
            using var lossFn = CrossEntropyLoss();
            var tLoss = lossFn.forward(tLogits, torchTarget);
            tLoss.backward();

            Console.WriteLine($"Loss: {tLoss.cpu().data<float>().ToArray()[0]:F6}");

            // Get gradients BEFORE update
            var torchConvGrad = torchConv.weight.grad.cpu().data<float>().ToArray();
            var torchDenseGrad = torchDense.weight.grad.cpu().data<float>().ToArray();
            Console.WriteLine($"\nTorchSharp Conv grad sum: {torchConvGrad.Sum():F6}");
            Console.WriteLine($"TorchSharp Dense grad sum: {torchDenseGrad.Sum():F6}");

            // Update weights
            using (torch.no_grad())
            {
                torchConv.weight.sub_(torchConv.weight.grad * learningRate);
                torchDense.weight.sub_(torchDense.weight.grad * learningRate);
            }

            // Get updated weights
            var torchConvWeightAfter = torchConv.weight.cpu().data<float>().ToArray();
            var torchDenseWeightAfter = torchDense.weight.cpu().data<float>().ToArray();
            Console.WriteLine($"TorchSharp Conv weight sum after update: {torchConvWeightAfter.Sum():F6}");
            Console.WriteLine($"TorchSharp Dense weight sum after update: {torchDenseWeightAfter.Sum():F6}");

            // ==================== NEURALNETS ====================
            Console.WriteLine("\n--- NeuralNets Setup ---");
            var inputShape = new InputOutputShape(28, 28, 1, 1);
            var conv1 = new ConvolutionLayer(inputShape, kernelCount: 5, kernelSquareDimension: 4, stride: 1);
            var relu1 = new ReLUActivaction(conv1.OutputShape);
            var pool1 = new PoolingLayer(relu1.OutputShape, stride: 2, kernelCount: 5, kernelSquareDimension: 2, kernelDepth: 1);
            var flatten = new FlattenLayer(pool1.OutputShape, nodeCount: 1);
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 10);
            var softmax = new SoftMax(dense.OutputShape, nodeCount: 10);

            // Note: No SoftMax layer - we compare raw logits like TorchSharp
            var layers = new List<Layer> { conv1, relu1, pool1, flatten, dense };

            // Zero initialization
            for (int k = 0; k < 5; k++)
            {
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        conv1.Kernels[k, 0][r, c] = 0;
                conv1.Biases[k][0, 0] = 0;
            }
            for (int r = 0; r < 10; r++)
                for (int c = 0; c < 720; c++)
                    dense.Weights[r, c] = 0;
            dense.Biases.SetRandom(42, 0, 0);

            var network = new GeneralFeedForwardANN(layers, learningRate, 28 * 28, 10, new CategoricalCrossEntropy());

            // Reset accumulators
            foreach (var layer in layers)
                layer.ResetAccumulators();

            // Forward
            MatrixLibrary.Tensor nnOutput = pair.Input;
            Console.WriteLine($"\nForward pass layer by layer:");
            Console.WriteLine($"  Initial input: Matrices={nnOutput?.Matrices?.Count}, ColumnVector={nnOutput?.ToColumnVector()?.Size}");
            int layerIdx = 0;
            foreach (var layer in layers)
            {
                try
                {
                    string beforeShape;
                    if (nnOutput?.ToColumnVector() != null)
                        beforeShape = $"vec({nnOutput.ToColumnVector().Size})";
                    else if (nnOutput?.Matrices != null && nnOutput.Matrices.Count > 0 && nnOutput.Matrices[0] != null)
                        beforeShape = $"{nnOutput.Matrices[0].Rows}x{nnOutput.Matrices[0].Cols}x{nnOutput.Matrices.Count}";
                    else
                        beforeShape = "unknown";
                    
                    nnOutput = layer.FeedFoward(nnOutput);
                    
                    string afterShape;
                    if (nnOutput?.ToColumnVector() != null)
                        afterShape = $"vec({nnOutput.ToColumnVector().Size})";
                    else if (nnOutput?.Matrices != null && nnOutput.Matrices.Count > 0 && nnOutput.Matrices[0] != null)
                    {
                        // Debug: Print details for FlattenLayer input
                        if (layer is FlattenLayer)
                        {
                            Console.WriteLine($"    DEBUG FlattenLayer input: {nnOutput.Matrices.Count} matrices");
                            for (int m = 0; m < nnOutput.Matrices.Count && m < 3; m++)
                            {
                                var mat = nnOutput.Matrices[m];
                                Console.WriteLine($"      Matrix {m}: {mat?.Rows}x{mat?.Cols} (null: {mat == null})");
                            }
                        }
                        afterShape = $"{nnOutput.Matrices[0].Rows}x{nnOutput.Matrices[0].Cols}x{nnOutput.Matrices.Count}";
                    }
                    else
                        afterShape = "null";
                    
                    Console.WriteLine($"  Layer {layerIdx} ({layer.GetType().Name}): {beforeShape} -> {afterShape}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  ERROR at Layer {layerIdx} ({layer.GetType().Name}): {ex.Message}");
                    Console.WriteLine($"    nnOutput is null: {nnOutput == null}");
                    Console.WriteLine($"    nnOutput.Matrices is null: {nnOutput?.Matrices == null}");
                    if (nnOutput?.Matrices != null)
                        Console.WriteLine($"    nnOutput.Matrices.Count: {nnOutput.Matrices.Count}");
                    throw;
                }
                layerIdx++;
            }

            var nnPredicted = nnOutput.ToColumnVector();
            Assert.IsNotNull(nnPredicted);

            // Check forward outputs match
            var nnLogits = nnPredicted.Column;
            var torchLogits = tLogits.cpu().data<float>().ToArray();
            
            Console.WriteLine($"\nNeuralNets logits (first 5): [{string.Join(", ", nnLogits.Take(5).Select(x => x.ToString("F6")))}...]");
            
            float maxLogitDiff = 0;
            for (int i = 0; i < 10; i++)
            {
                float diff = System.Math.Abs(nnLogits[i] - torchLogits[i]);
                if (diff > maxLogitDiff) maxLogitDiff = diff;
            }
            Console.WriteLine($"Max logit difference: {maxLogitDiff:F6}");
            Assert.IsTrue(maxLogitDiff < Tolerance, $"Forward pass outputs should match. Max diff: {maxLogitDiff:F6}");

            // Loss
            float nnLoss = network.GetTotallLoss(pair, nnPredicted);
            float torchLoss = tLoss.cpu().data<float>().ToArray()[0];
            Console.WriteLine($"\nNeuralNets loss: {nnLoss:F6}");
            Console.WriteLine($"Loss difference: {System.Math.Abs(nnLoss - torchLoss):F6}");

            // Backward
            var lossDerivative = network.LossFunction.Derivative(pair.Output.ToColumnVector()!, nnPredicted);
            MatrixLibrary.Tensor dE_dX = lossDerivative.ToTensor();

            Console.WriteLine($"\nBackward pass:");
            layerIdx = layers.Count - 1;
            foreach (var layer in layers.Reverse<Layer>())
            {
                Console.WriteLine($"  Layer {layerIdx} ({layer.GetType().Name})");
                
                // Store gradient info before backprop
                if (layer is WeightedLayer wl && wl.LastWeightGradient != null)
                {
                    Console.WriteLine($"    Weight gradient shape: {wl.LastWeightGradient.Rows}x{wl.LastWeightGradient.Cols}");
                }
                if (layer is ConvolutionLayer cl && cl.KernelGradientAccumulator != null)
                {
                    Console.WriteLine($"    Kernel gradient count: {cl.KernelGradientAccumulator.Stacks.Count}");
                }
                
                dE_dX = layer.BackPropagation(dE_dX);
                layerIdx--;
            }

            // Check conv gradients
            float nnConvGradSum = 0;
            for (int k = 0; k < 5; k++)
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        nnConvGradSum += conv1.KernelGradientAccumulator[k, 0][r, c];

            Console.WriteLine($"\nNeuralNets Conv grad sum: {nnConvGradSum:F6}");
            Console.WriteLine($"Conv gradient difference: {System.Math.Abs(nnConvGradSum - torchConvGrad.Sum()):F6}");

            // Check dense gradients
            float nnDenseGradSum = 0;
            if (dense.LastWeightGradient != null)
            {
                for (int r = 0; r < 10; r++)
                    for (int c = 0; c < 720; c++)
                        nnDenseGradSum += dense.LastWeightGradient[r, c];
            }

            Console.WriteLine($"NeuralNets Dense grad sum: {nnDenseGradSum:F6}");
            Console.WriteLine($"Dense gradient difference: {System.Math.Abs(nnDenseGradSum - torchDenseGrad.Sum()):F6}");

            // Update weights
            foreach (var layer in layers)
                layer.UpdateWeightsAndBiasesWithScaledGradients(learningRate);

            // Check updated weights
            float nnConvWeightSum = 0;
            for (int k = 0; k < 5; k++)
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        nnConvWeightSum += conv1.Kernels[k, 0][r, c];

            float nnDenseWeightSum = 0;
            for (int r = 0; r < 10; r++)
                for (int c = 0; c < 720; c++)
                    nnDenseWeightSum += dense.Weights[r, c];

            Console.WriteLine($"\nNeuralNets Conv weight sum after update: {nnConvWeightSum:F6}");
            Console.WriteLine($"NeuralNets Dense weight sum after update: {nnDenseWeightSum:F6}");

            Console.WriteLine($"\nWeight update difference (Conv): {System.Math.Abs(nnConvWeightSum - torchConvWeightAfter.Sum()):F6}");
            Console.WriteLine($"Weight update difference (Dense): {System.Math.Abs(nnDenseWeightSum - torchDenseWeightAfter.Sum()):F6}");

            // Verify gradients match
            float convGradDiff = System.Math.Abs(nnConvGradSum - torchConvGrad.Sum());
            float denseGradDiff = System.Math.Abs(nnDenseGradSum - torchDenseGrad.Sum());

            Console.WriteLine("\n=== Summary ===");
            Console.WriteLine($"Forward pass: {(maxLogitDiff < Tolerance ? "✓ MATCH" : "✗ DIFFER")}");
            Console.WriteLine($"Conv gradients: {(convGradDiff < Tolerance ? "✓ MATCH" : "✗ DIFFER")} (diff: {convGradDiff:F6})");
            Console.WriteLine($"Dense gradients: {(denseGradDiff < Tolerance ? "✓ MATCH" : "✗ DIFFER")} (diff: {denseGradDiff:F6})");
        }

        private static int ArgMax(float[] arr)
        {
            int maxIdx = 0;
            float maxVal = arr[0];
            for (int i = 1; i < arr.Length; i++)
            {
                if (arr[i] > maxVal)
                {
                    maxVal = arr[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }
    }
}
