using System.Diagnostics;
using MatrixLibrary;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MnistReader_ANN;
using NeuralNets;
using NeuralNetsTests;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace NeuralNetsTests.torchSharpComparison
{
    /// <summary>
    /// Tests to verify CNN training convergence matches between NeuralNets and PyTorch/TorchSharp
    /// Using identical: topology, initialization, learning rate, optimizer (vanilla SGD), and data
    /// </summary>
    [TestClass]
    public class CNNConvergenceTests
    {
        private const float Tolerance = 0.1f;
        private const int RandomSeed = 42;
        private const int numEpochs = 2;
        private const float learningRate = 0.001f;
        private const int batchSize = 64;

        /// <summary>
        /// Compares CNN training convergence between NeuralNets and TorchSharp
        /// Both use: Conv(4x4,5) → ReLU → Pool(2x2) → Flatten → Dense(720→10) → SoftMax
        /// Both use: Vanilla SGD, learning_rate=0.001, 5 epochs, same Xavier init, same 100 MNIST samples
        /// </summary>
        [TestMethod]
        [Ignore]
        public void CNN_VanillaSGD_ConvergenceComparison()
        {
            Console.WriteLine("\n=== CNN Vanilla SGD Convergence Comparison ===");
            Console.WriteLine("Architecture: Conv(4x4,5) → ReLU → Pool(2x2) → Flatten → Dense(720→10) → SoftMax");
            Console.WriteLine("Optimizer: Vanilla SGD (no momentum)");
            Console.WriteLine($"Learning Rate: {CNNConvergenceTests.learningRate}, Epochs: {CNNConvergenceTests.numEpochs}, Samples: 100\n");

            // Get same MNIST samples for both networks
            var trainingSet = new MNISTTrainingSet();
            var trainingPairs = trainingSet.BuildNewRandomizedTrainingList(do2DImage: true).Take(1000).ToList();
            Console.WriteLine($"Loaded {trainingPairs.Count} training samples\n");

            // ==================== TORCHSHARP CNN ====================
            Console.WriteLine("--- Training TorchSharp CNN ---");
            var (torchFinalLoss, torchFinalAccuracy) = TrainTorchSharpCNN(
                trainingPairs, 
                epochs: CNNConvergenceTests.numEpochs, 
                learningRate: CNNConvergenceTests.learningRate);

            // ==================== NEURALNETS CNN ====================
            Console.WriteLine("\n--- Training NeuralNets CNN ---");
            var (nnFinalLoss, nnFinalAccuracy) = TrainNeuralNetsCNN(
                trainingSet,
                trainingPairs, 
                epochs: CNNConvergenceTests.numEpochs, 
                learningRate: CNNConvergenceTests.learningRate);

            // ==================== COMPARE RESULTS ====================
            Console.WriteLine("\n=== Final Results Comparison ===");
            Console.WriteLine($"TorchSharp:  Loss={torchFinalLoss:F4}, Accuracy={torchFinalAccuracy:P2}");
            Console.WriteLine($"NeuralNets:  Loss={nnFinalLoss:F4}, Accuracy={nnFinalAccuracy:P2}");
            Console.WriteLine($"Loss Diff:   {System.Math.Abs(torchFinalLoss - nnFinalLoss):F4}");
            Console.WriteLine($"Accuracy Diff: {System.Math.Abs(torchFinalAccuracy - nnFinalAccuracy):P2}");

            // Both should show convergence (accuracy should increase, loss should decrease)
            Console.WriteLine("\n=== Convergence Check ===");
            bool torchConverged = torchFinalAccuracy > 0.15f;  // Better than random (10%)
            bool nnConverged = nnFinalAccuracy > 0.15f;

            Console.WriteLine($"TorchSharp converged: {torchConverged} (accuracy > 15%)");
            Console.WriteLine($"NeuralNets converged: {nnConverged} (accuracy > 15%)");

            // If TorchSharp converges but NeuralNets doesn't, we have a bug
            if (torchConverged && !nnConverged)
            {
                Assert.Fail($"TorchSharp converged (accuracy={torchFinalAccuracy:P2}) but NeuralNets did not (accuracy={nnFinalAccuracy:P2}). " +
                           "This indicates a bug in NeuralNets training implementation.");
            }

            // If both converge, great! If neither converges, might be hyperparameter issue
            if (torchConverged && nnConverged)
            {
                Console.WriteLine("\n✓ Both networks converged successfully!");
                
                // Verify they're within tolerance of each other
                float accuracyDiff = System.Math.Abs(torchFinalAccuracy - nnFinalAccuracy);
                if (accuracyDiff > 0.1f)  // Within 10% accuracy of each other
                {
                    Console.WriteLine($"⚠ Warning: Accuracy difference ({accuracyDiff:P2}) is > 10%");
                }
            }
            else if (!torchConverged && !nnConverged)
            {
                Console.WriteLine("\n⚠ Neither network converged - hyperparameters may need adjustment");
                Console.WriteLine("  Try: lower learning rate, more epochs, or mini-batching");
            }
        }

        /// <summary>
        /// Train TorchSharp CNN with vanilla SGD
        /// </summary>
        private static (float finalLoss, float finalAccuracy) TrainTorchSharpCNN(
            List<TrainingPair> trainingPairs, int epochs, float learningRate)
        {
            // Create network: Conv(4x4, 5 kernels) → ReLU → Pool(2x2) → Flatten → Dense(720→10)
            var torchConv = Conv2d((long)1, (long)5, (long)4, stride: (long)1, padding: (long)0);
            var torchPool = torch.nn.MaxPool2d((long)2, stride: (long)2);
            var torchDense = Linear(720, 10);  // 12*12*5 = 720

            // Xavier initialization to match NeuralNets
            float convBound = (float)System.Math.Sqrt(6.0 / (16 + 5));
            var convWeights = new float[5 * 1 * 4 * 4];  // out_channels * in_channels * kernel_h * kernel_w
            var rand = new Random(42);
            for (int i = 0; i < convWeights.Length; i++)
                convWeights[i] = (float)(rand.NextDouble() * 2 * convBound - convBound);
            torchConv.weight = torch.from_array(convWeights).reshape(5, 1, 4, 4).AsParameter();
            torchConv.bias = torch.zeros(5).AsParameter();

            float denseBound = (float)System.Math.Sqrt(6.0 / (720 + 10));
            var denseWeights = new float[10 * 720];
            for (int i = 0; i < denseWeights.Length; i++)
                denseWeights[i] = (float)(rand.NextDouble() * 2 * denseBound - denseBound);
            torchDense.weight = torch.from_array(denseWeights).reshape(10, 720).AsParameter();
            torchDense.bias = torch.zeros(10).AsParameter();

            float finalLoss = 0;
            float finalAccuracy = 0;

            // Training loop
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float epochLoss = 0;
                int correct = 0;
                int total = 0;

                foreach (var pair in trainingPairs)
                {
                    // Prepare input
                    var inputData = pair.Input.Matrices[0].Mat;
                    using var torchInput = torch.from_array(inputData).reshape(1, 1, 28, 28);
                    
                    // Prepare target
                    int trueLabel = ArgMax(pair.Output.ToColumnVector()!.Column);
                    using var torchTarget = torch.from_array(new long[] { trueLabel });

                    // Zero gradients (vanilla SGD - no momentum)
                    torchConv.zero_grad();
                    torchDense.zero_grad();

                    // Forward pass
                    var convOut = torchConv.forward(torchInput);
                    var reluOut = relu(convOut);
                    var poolOut = torchPool.forward(reluOut);
                    var flatOut = poolOut.reshape(1, 720);
                    var logits = torchDense.forward(flatOut);

                    // Compute loss
                    using var lossFn = CrossEntropyLoss();
                    var loss = lossFn.forward(logits, torchTarget);
                    
                    epochLoss += loss.cpu().data<float>().ToArray()[0];

                    // Track accuracy
                    var pred = logits.argmax(1).cpu().data<long>().ToArray()[0];
                    if (pred == trueLabel) correct++;
                    total++;

                    // Backward pass
                    loss.backward();

                    // Manual SGD update (vanilla - no momentum)
                    using (torch.no_grad())
                    {
                        torchConv.weight.sub_(torchConv.weight.grad * learningRate);
                        torchConv.bias.sub_(torchConv.bias.grad * learningRate);
                        torchDense.weight.sub_(torchDense.weight.grad * learningRate);
                        torchDense.bias.sub_(torchDense.bias.grad * learningRate);
                    }
                }

                finalLoss = epochLoss / total;
                finalAccuracy = (float)correct / total;
                Console.WriteLine($"Epoch {epoch,2}: Loss={finalLoss:F4}, Accuracy={finalAccuracy:P2} ({correct}/{total})");
            }

            return (finalLoss, finalAccuracy);
        }

        /// <summary>
        /// Train NeuralNets CNN with vanilla SGD (manual loop - single sample)
        /// </summary>
        private static (float finalLoss, float finalAccuracy) TrainNeuralNetsCNN(
            ITrainingSet trainingSet,
            List<TrainingPair> trainingPairs,
            int epochs,
            float learningRate)
        {
            MatrixFactory.DefaultBackend = MatrixBackend.AVX;
            // Create network: Conv(4x4, 5) → ReLU → Pool(2x2) → Flatten → Dense(720→10) → SoftMax
            // Using SoftMax layer + CategoricalCrossEntropy like CNNTrainingTests
            var inputShape = new InputOutputShape(28, 28, 1, 1);
            var conv1 = new ConvolutionLayer(inputShape, kernelCount: 5, kernelSquareDimension: 4, stride: 1);
            var relu1 = new ReLUActivaction(conv1.OutputShape);
            var pool1 = new PoolingLayer(relu1.OutputShape, stride: 2, kernelCount: 5, kernelSquareDimension: 2, kernelDepth: 1);
            var flatten = new FlattenLayer(pool1.OutputShape, nodeCount: 1);
            var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 10);
            var softmax = new SoftMax(dense.OutputShape, nodeCount: 10);

            var layers = new List<Layer> { conv1, relu1, pool1, flatten, dense, softmax };

            // Xavier initialization (same as TorchSharp)
            float convBound = (float)System.Math.Sqrt(6.0 / (16 + 5));
            var rand = new Random(42);
            for (int k = 0; k < 5; k++)
            {
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        conv1.Kernels[k, 0][r, c] = (float)(rand.NextDouble() * 2 * convBound - convBound);
                conv1.Biases[k][0, 0] = 0;
            }

            float denseBound = (float)System.Math.Sqrt(6.0 / (720 + 10));
            for (int r = 0; r < 10; r++)
                for (int c = 0; c < 720; c++)
                    dense.Weights[r, c] = (float)(rand.NextDouble() * 2 * denseBound - denseBound);
            dense.Biases.SetRandom(42, -0.1f, 0.1f);

            var network = new GeneralFeedForwardANN(
                layers, learningRate, inputDim: 28 * 28, outputDim: 10,
                new CategoricalCrossEntropy());

            float finalLoss = 0;
            float finalAccuracy = 0;

            // Training loop (vanilla SGD - no momentum, single sample updates)
            var ctx = new ConvolutionRenderContext(network, CNNConvergenceTests.batchSize, trainingSet);
            ctx.EpochTrain(epochs);

            finalAccuracy = RunNetworkOnMnistTestSet(network, ctx);

            return (finalLoss, finalAccuracy);
        }

        private static float RunNetworkOnMnistTestSet(GeneralFeedForwardANN network, ConvolutionRenderContext ctx)
        {
            MNISTTrainingSet trainingSet = new MNISTTrainingSet();
            var testSet = trainingSet.GetTestPairs(true);
            int totalSamples = 0;
            int totalCorrectSamples = 0;
            foreach (TrainingPair testPair in testSet)
            {
                // run actual vs expected
                var predicted = ctx.FeedForward(testPair.Input);
                var sm = CategoricalCrossEntropy.SoftMax(predicted.ToColumnVector()); // just so i can see the probability distribution.
                var oneHotPredicted = OneHotEncode(predicted.ToColumnVector().Column);
                var expected = testPair.Output;

                if (IsSamePrediction(oneHotPredicted, expected.ToColumnVector().Column))
                {
                    totalCorrectSamples++;
                }
                totalSamples++;
            }

            float acc = (float)totalCorrectSamples / totalSamples;
            Console.WriteLine($"Total Samples: {totalSamples}, Total Correct: {totalCorrectSamples}, Accuracy: {acc}");
            return acc;
        }

        private static bool IsSamePrediction(float[] a, float[] b)
        {
            Debug.Assert(a.Length == b.Length);
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                {
                    return false;
                }
            }
            return true;
        }

        public static float[] OneHotEncode(float[] logits)
        {
            // 1. Find the index of the highest value (Argmax)
            int maxIndex = 0;
            float maxValue = logits[0];

            for (int i = 1; i < logits.Length; i++)
            {
                if (logits[i] > maxValue)
                {
                    maxValue = logits[i];
                    maxIndex = i;
                }
            }

            // 2. Create a one-hot array of the same length
            float[] oneHot = new float[logits.Length];
            oneHot[maxIndex] = 1.0f;

            return oneHot;
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
