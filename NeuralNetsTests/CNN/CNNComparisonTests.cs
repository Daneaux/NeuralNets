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
    [TestClass]
    public class CNNComparisonTests
    {
        private const float Tolerance = 1e-3f;
        private const int RandomSeed = 42;

        /// <summary>
        /// Simple CNN comparison test
        /// Architecture: 28x28 -> Conv(3x3, 1 kernel) -> ReLU -> Dense(10) -> SoftMax
        /// This minimal architecture makes it easier to debug issues.
        /// </summary>
        [TestMethod]
        public void SimpleCNN_LayerByLayer_Comparison()
        {
            Console.WriteLine("\n=== Simple CNN Layer-by-Layer Comparison ===\n");

            // Get a single MNIST sample (2D image)
            var trainingSet = new MNISTTrainingSet();
            var sample = trainingSet.BuildNewRandomizedTrainingList(do2DImage: true).First();
            
            Console.WriteLine($"Input shape: 28x28x1");
            Console.WriteLine($"Sample label: {ArgMax(sample.Output.ToColumnVector().Column)}\n");

            // Extract 2D input data
            var input2D = sample.Input.Matrices[0]; // 28x28 matrix
            float[,] inputData = input2D.Mat;
            
            // ==================== TORCHSHARP CNN ====================
            Console.WriteLine("--- TorchSharp CNN ---");
            
            // Create simple CNN: Conv(3x3, 1 kernel) -> ReLU -> Flatten -> Dense(10)
            // Explicitly cast to long to disambiguate overloads
            var torchConv = Conv2d((long)1, (long)1, (long)3, stride: (long)1, padding: (long)0);
            var torchDense = Linear(26 * 26, 10);  // After 3x3 conv on 28x28 with no padding: 26x26
            
            // Set weights to known values for comparison
            float[] convWeights = new float[9]; // 3x3 kernel
            for (int i = 0; i < 9; i++) convWeights[i] = (float)(i + 1) / 10.0f; // 0.1, 0.2, ..., 0.9
            using var torchConvWeight = torch.from_array(convWeights).reshape(1, 1, 3, 3);
            torchConv.weight = torchConvWeight.AsParameter();
            torchConv.bias = torch.zeros(1).AsParameter();
            
            float[] denseWeights = new float[26 * 26 * 10];
            for (int i = 0; i < denseWeights.Length; i++) denseWeights[i] = (float)(i % 10) / 100.0f;
            using var torchDenseWeight = torch.from_array(denseWeights).reshape(10, 26 * 26);
            torchDense.weight = torchDenseWeight.AsParameter();
            torchDense.bias = torch.zeros(10).AsParameter();
            
            // Forward pass
            using var torchInput = torch.from_array(inputData).reshape(1, 1, 28, 28);
            var torchConvOut = torchConv.forward(torchInput);
            var torchReLUOut = relu(torchConvOut);
            var torchFlatOut = torchReLUOut.reshape(1, 26 * 26);
            var torchDenseOut = torchDense.forward(torchFlatOut);
            var torchSoftmaxOut = softmax(torchDenseOut, dim: 1);
            
            var torchOutput = torchSoftmaxOut.cpu().data<float>().ToArray();
            Console.WriteLine($"TorchSharp output (first 5): [{string.Join(", ", torchOutput.Take(5).Select(x => x.ToString("F6")))}...]");
            Console.WriteLine($"TorchSharp prediction: {ArgMax(torchOutput)}\n");

            // ==================== NEURALNETS CNN ====================
            Console.WriteLine("--- NeuralNets CNN ---");
            
            // Build equivalent CNN
            var inputShape = new InputOutputShape(28, 28, 1, 1);
            
            // Conv layer: 1 kernel, 3x3, stride 1
            var conv1 = new ConvolutionLayer(inputShape, kernelCount: 1, kernelSquareDimension: 3, stride: 1);
            // Set weights to match TorchSharp
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    conv1.Kernels[0, 0][r, c] = convWeights[r * 3 + c];
            // Bias is a 1x1 matrix, set its value using SetRandom
            conv1.Biases[0].SetRandom(42, 0, 0);
            
            // ReLU
            var relu1 = new ReLUActivaction(conv1.OutputShape);
            
            // Flatten: 26x26x1 = 676 nodes
            var flatten = new FlattenLayer(relu1.OutputShape, nodeCount: 1);
            
            // Dense: 676 -> 10
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 10);
            // Set weights to match TorchSharp
            for (int r = 0; r < 10; r++)
                for (int c = 0; c < 26 * 26; c++)
                    dense.Weights[r, c] = denseWeights[r * 26 * 26 + c];
            // Biases is a ColumnVector, use SetRandom to set all to 0
            dense.Biases.SetRandom(42, 0, 0);
            
            // SoftMax
            var softmax1 = new SoftMax(dense.OutputShape, nodeCount: 10);
            
            // Build network
            var layers = new List<Layer> { conv1, relu1, flatten, dense, softmax1 };
            var network = new GeneralFeedForwardANN(layers, 0.01f, 28 * 28, 10, new CategoricalCrossEntropy());
            
            // Forward pass layer by layer for debugging
            Console.WriteLine("\nLayer-by-layer forward pass:");
            
            // Input
            MatrixLibrary.Tensor current = sample.Input;
            Console.WriteLine($"  Input: {current.Matrices[0].Rows}x{current.Matrices[0].Cols}x{current.Matrices.Count}");
            
            // Conv
            current = conv1.FeedFoward(current);
            var convOutVec = current.ToColumnVector();
            if (convOutVec != null)
            {
                Console.WriteLine($"  Conv output (first 5): [{string.Join(", ", Enumerable.Range(0, 5).Select(i => convOutVec[i].ToString("F6")))}...]");
            }
            else if (current.Matrices != null)
            {
                Console.WriteLine($"  Conv output: {current.Matrices[0].Rows}x{current.Matrices[0].Cols}x{current.Matrices.Count} matrix");
                Console.WriteLine($"    First 5 values: [{string.Join(", ", Enumerable.Range(0, 5).Select(i => current.Matrices[0][i / current.Matrices[0].Cols, i % current.Matrices[0].Cols].ToString("F6")))}...]");
            }
            
            // ReLU
            current = relu1.FeedFoward(current);
            Console.WriteLine($"  ReLU output shape: {current.Matrices[0].Rows}x{current.Matrices[0].Cols}x{current.Matrices.Count}");
            
            // Flatten
            current = flatten.FeedFoward(current);
            var flatVec = current.ToColumnVector();
            Console.WriteLine($"  Flatten output size: {flatVec?.Size}");
            Console.WriteLine($"    First 5: [{string.Join(", ", Enumerable.Range(0, 5).Select(i => flatVec[i].ToString("F6")))}...]");
            
            // Dense
            current = dense.FeedFoward(current);
            var denseVec = current.ToColumnVector();
            Console.WriteLine($"  Dense output size: {denseVec?.Size}");
            Console.WriteLine($"    Values: [{string.Join(", ", Enumerable.Range(0, 10).Select(i => denseVec[i].ToString("F6")))}]");
            
            // SoftMax
            current = softmax1.FeedFoward(current);
            var nnOutput = current.ToColumnVector().Column;
            
            Console.WriteLine($"\nNeuralNets output (first 5): [{string.Join(", ", nnOutput.Take(5).Select(x => x.ToString("F6")))}...]");
            Console.WriteLine($"NeuralNets prediction: {ArgMax(nnOutput)}\n");

            // ==================== COMPARE OUTPUTS ====================
            Console.WriteLine("--- Comparison ---");
            float maxDiff = 0;
            for (int i = 0; i < 10; i++)
            {
                float diff = System.Math.Abs(torchOutput[i] - nnOutput[i]);
                if (diff > maxDiff) maxDiff = diff;
                if (diff > Tolerance)
                {
                    Console.WriteLine($"  MISMATCH at index {i}: Torch={torchOutput[i]:F6}, NN={nnOutput[i]:F6}, Diff={diff:F6}");
                }
            }
            
            Console.WriteLine($"\nMax difference: {maxDiff:F6}");
            Console.WriteLine($"Tolerance: {Tolerance:F6}");
            
            Assert.IsTrue(maxDiff < Tolerance, 
                $"CNN outputs should match within tolerance. Max diff: {maxDiff:F6}");
            
            Console.WriteLine("\n✓ CNN outputs match!");
        }

        /// <summary>
        /// Compare gradients after single training step
        /// </summary>
        [TestMethod]
        public void SimpleCNN_GradientComparison()
        {
            Console.WriteLine("\n=== Simple CNN Gradient Comparison ===\n");

            // Get a single MNIST sample
            var trainingSet = new MNISTTrainingSet();
            var sample = trainingSet.BuildNewRandomizedTrainingList(do2DImage: true).First();
            var input2D = sample.Input.Matrices[0];
            float[,] inputData = input2D.Mat;

            // Target label
            int targetLabel = ArgMax(sample.Output.ToColumnVector().Column);

            // ==================== TORCHSHARP ====================
            Console.WriteLine("--- TorchSharp ---");

            var torchConv = Conv2d((long)1, (long)1, (long)3, stride: (long)1, padding: (long)0);
            var torchDense = Linear(26 * 26, 10);

            // Initialize with same seed as NeuralNets for comparison
            // For now, use zeros for simplicity
            torchConv.weight = torch.zeros(1, 1, 3, 3).AsParameter();
            torchConv.bias = torch.zeros(1).AsParameter();
            torchDense.weight = torch.zeros(10, 26 * 26).AsParameter();
            torchDense.bias = torch.zeros(10).AsParameter();

            using var torchInput = torch.from_array(inputData).reshape(1, 1, 28, 28);
            // Target needs to be a 1D tensor with batch size 1
            long[] targetArray = new long[] { targetLabel };
            using var torchTarget = torch.from_array(targetArray);  // Shape: [1]

            // Forward + backward
            var torchConvOut = torchConv.forward(torchInput);
            var torchReLUOut = relu(torchConvOut);
            var torchFlatOut = torchReLUOut.reshape(1, 26 * 26);
            var torchDenseOut = torchDense.forward(torchFlatOut);

            using var torchLossFn = CrossEntropyLoss();
            var loss = torchLossFn.forward(torchDenseOut, torchTarget);
            loss.backward();

            // Get gradients
            var torchConvGrad = torchConv.weight.grad.cpu().data<float>().ToArray();
            var torchConvBiasGrad = torchConv.bias.grad.cpu().data<float>().ToArray();
            var torchDenseGrad = torchDense.weight.grad.cpu().data<float>().ToArray();
            var torchDenseBiasGrad = torchDense.bias.grad.cpu().data<float>().ToArray();

            Console.WriteLine($"TorchSharp loss: {loss.cpu().data<float>().ToArray()[0]:F6}");
            Console.WriteLine($"TorchSharp conv weight grad: [{string.Join(", ", torchConvGrad.Take(9).Select(x => x.ToString("F6")))}]");
            Console.WriteLine($"TorchSharp dense weight grad (first 5): [{string.Join(", ", torchDenseGrad.Take(5).Select(x => x.ToString("F6")))}...]");

            // ==================== NEURALNETS ====================
            Console.WriteLine("\n--- NeuralNets ---");

            var inputShape = new InputOutputShape(28, 28, 1, 1);
            var conv1 = new ConvolutionLayer(inputShape, kernelCount: 1, kernelSquareDimension: 3, stride: 1);
            var relu1 = new ReLUActivaction(conv1.OutputShape);
            var flatten = new FlattenLayer(relu1.OutputShape, nodeCount: 1);
            var annWeightedLayer = new WeightedLayer(flatten.OutputShape, nodeCount: 10);
            //var softmax1 = new SoftMax(annWeightedLayer.OutputShape, nodeCount: 10);

            // Zero initialize to match TorchSharp
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    conv1.Kernels[0, 0][r, c] = 0;
            conv1.Biases[0][0, 0] = 0;  // Set bias value at [0,0] since it's a matrix

            for (int r = 0; r < 10; r++)
                for (int c = 0; c < 26 * 26; c++)
                    annWeightedLayer.Weights[r, c] = 0;
            annWeightedLayer.Biases.SetRandom(42, 0, 0);  // Set all biases to 0

            var layers = new List<Layer> { conv1, relu1, flatten, annWeightedLayer };//, softmax1 };
            var network = new GeneralFeedForwardANN(layers, 0.01f, 28 * 28, 10, new CategoricalCrossEntropy());

            // Reset accumulators
            foreach (var layer in layers)
                layer.ResetAccumulators();

            // Forward pass
            MatrixLibrary.Tensor output = sample.Input;
            foreach (var layer in layers)
            {
                output = layer.FeedFoward(output);
            }
            var predicted = output.ToColumnVector();

            // Calculate loss
            float nnLoss = network.GetTotallLoss(sample, predicted);
            Console.WriteLine($"NeuralNets loss: {nnLoss:F6}");

            // Backward pass
            var lossDerivative = network.LossFunction.Derivative(
                sample.Output.ToColumnVector(), predicted);
            MatrixLibrary.Tensor dE_dX = lossDerivative.ToTensor();

            foreach (var layer in layers.Reverse<Layer>())
            {
                dE_dX = layer.BackPropagation(dE_dX);
            }

            // Check conv layer gradients
            Console.WriteLine($"\n--- Conv Layer Gradients ---");
            if (conv1.KernelGradientAccumulator != null && conv1.KernelGradientAccumulator.Stacks.Count > 0)
            {
                var nnConvGrad = new float[9];
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        nnConvGrad[r * 3 + c] = conv1.KernelGradientAccumulator[0, 0][r, c];

                Console.WriteLine($"NeuralNets conv weight grad: [{string.Join(", ", nnConvGrad.Select(x => x.ToString("F6")))}]");

                // Compare conv gradients
                float maxConvGradDiff = 0;
                for (int i = 0; i < System.Math.Min(torchConvGrad.Length, nnConvGrad.Length); i++)
                {
                    float diff = System.Math.Abs(torchConvGrad[i] - nnConvGrad[i]);
                    if (diff > maxConvGradDiff) maxConvGradDiff = diff;
                    if (diff > Tolerance)
                    {
                        Console.WriteLine($"  Conv grad mismatch at [{i}]: Torch={torchConvGrad[i]:F6}, NN={nnConvGrad[i]:F6}, Diff={diff:F6}");
                    }
                }

                Console.WriteLine($"Max conv gradient difference: {maxConvGradDiff:F6}");

                // Assert conv gradients match
                Assert.IsTrue(maxConvGradDiff < Tolerance,
                    $"Conv gradients should match within tolerance. Max diff: {maxConvGradDiff:F6}");

                Console.WriteLine("✓ Conv gradients match!");
            }
            else
            {
                Assert.Fail("Conv layer gradients were not computed");
            }

            // Check dense layer gradients
            Console.WriteLine($"\n--- Dense Layer Weight Gradients ---");
            Console.WriteLine($"NeuralNets dense weight grad shape: {annWeightedLayer.LastWeightGradient?.Rows}x{annWeightedLayer.LastWeightGradient?.Cols}");
            Assert.IsNotNull(annWeightedLayer.LastWeightGradient, "Dense layer weight gradients should not be null");

            var nnDenseGrad = MatrixHelpers.FlattenMatrixToFloatArray(annWeightedLayer.LastWeightGradient);
            Console.WriteLine($"NeuralNets dense weight grad (first 5): [{string.Join(", ", nnDenseGrad.Take(5).Select(x => x.ToString("F6")))}...]");

            // ---
            // Compare Weight gradients
            // ---
            float maxGradDiff = 0;
            Assert.AreEqual(nnDenseGrad.Length, torchDenseGrad.Length, "Dense gradient arrays should have the same length");
            for (int i = 0; i < System.Math.Min(torchDenseGrad.Length, nnDenseGrad.Length); i++)
            {
                float diff = System.Math.Abs(torchDenseGrad[i] - nnDenseGrad[i]);
                if (diff > maxGradDiff) maxGradDiff = diff;
            }

            Console.WriteLine($"Max dense gradient difference: {maxGradDiff:F6}");
            Console.WriteLine($"Tolerance: {Tolerance:F6}");

            // Assert gradients match
            Assert.IsTrue(maxGradDiff < Tolerance, $"Dense gradients should match within tolerance. Max diff: {maxGradDiff:F6}");
            Console.WriteLine("\n✓ Dense gradients match!");


            // ---
            // Check Biases on weighted layer
            // ---
            Console.WriteLine($"\n--- Dense Layer Bias Gradients ---");
            Assert.IsNotNull(annWeightedLayer.LastBiasGradient, "Dense layer bias gradients should not be null");
            Assert.AreEqual(torchDenseBiasGrad.Length, annWeightedLayer.LastBiasGradient.Size, "Dense bias gradient size should match TorchSharp");
            float maxBiasGradDiff = 0;
            for (int i = 0; i < torchDenseBiasGrad.Length; i++)
            {
                float diff = System.Math.Abs(torchDenseBiasGrad[i] - annWeightedLayer.LastBiasGradient[i]);
                if (diff > maxBiasGradDiff) maxBiasGradDiff = diff;
            }
            Assert.IsTrue(maxBiasGradDiff < Tolerance, $"Dense bias gradients should match within tolerance. Max diff: {maxBiasGradDiff:F6}");
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
