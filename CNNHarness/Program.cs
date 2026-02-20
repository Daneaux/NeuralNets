// CNNHarness - Convolutional Neural Network for MNIST Digit Recognition
using MnistReader_ANN;
using MatrixLibrary;
using NeuralNets;
using System.Diagnostics;

class CNNHarness
{
    static int Main(String[] args)
    {
        Console.WriteLine("=== CNN MNIST Digit Recognition ===\n");
        
        // Use GPU for matrix operations if available
        MatrixFactory.SetDefaultBackend(MatrixBackend.AVX);
        
        // Train the CNN with mini-batching (batch size 64 for stable convergence)
        Console.WriteLine("Training CNN with mini-batch size 64...");
        (var network, var layers) = TrainCNN(epochs: 30, learningRate: 0.001f, batchSize: 64);
        
        // Evaluate on test set
        Console.WriteLine("\nEvaluating on test set...");
        RunNetworkOnMnistTestSet(network, layers);

        return 0;
    }

    /// <summary>
    /// Evaluates the trained CNN on the MNIST test set
    /// </summary>
    private static void RunNetworkOnMnistTestSet(GeneralFeedForwardANN network, List<Layer> layers)
    {
        MNISTTrainingSet trainingSet = new MNISTTrainingSet();
        // Get test pairs in 2D format for CNN
        var testSet = trainingSet.GetTestPairs(do2DImage: true);
        int totalSamples = 0;
        int totalCorrectSamples = 0;
        
        foreach (TrainingPair testPair in testSet)
        {
            // Manual forward pass through all layers
            Tensor output = testPair.Input;
            foreach (var layer in layers)
            {
                output = layer.FeedFoward(output);
            }
            var predicted = output.ToColumnVector();
            if (predicted == null) continue;
            
            // Apply softmax to get probabilities (for visualization)
            var sm = CategoricalCrossEntropy.SoftMax(predicted);
            
            // Convert to one-hot encoding for comparison
            var oneHotPredicted = OneHotEncode(predicted.Column);
            var expected = testPair.Output;
            var expectedCol = expected.ToColumnVector();
            if (expectedCol == null) continue;
            
            if (IsSamePrediction(oneHotPredicted, expectedCol.Column))
            {
                totalCorrectSamples++;
            }
            totalSamples++;
            
            // Print progress every 1000 samples
            if (totalSamples % 1000 == 0)
            {
                float currentAccuracy = (float)totalCorrectSamples / totalSamples;
                Console.WriteLine($"  Processed {totalSamples} samples... Current accuracy: {currentAccuracy:P2}");
            }
        }
        
        float finalAccuracy = (float)totalCorrectSamples / totalSamples;
        Console.WriteLine($"\n=== Final Results ===");
        Console.WriteLine($"Total Samples: {totalSamples}");
        Console.WriteLine($"Total Correct: {totalCorrectSamples}");
        Console.WriteLine($"Accuracy: {finalAccuracy:P2} ({finalAccuracy*100:F1}%)");
    }

    /// <summary>
    /// Compares two one-hot encoded vectors
    /// </summary>
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

    /// <summary>
    /// Converts logits to one-hot encoding using argmax
    /// </summary>
    public static float[] OneHotEncode(float[] logits)
    {
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

        float[] oneHot = new float[logits.Length];
        oneHot[maxIndex] = 1.0f;

        return oneHot;
    }

    /// <summary>
    /// Applies proper Xavier/Glorot initialization to a weighted layer
    /// </summary>
    private static void XavierInitialize(WeightedLayer layer, int fanIn, int fanOut, int seed)
    {
        float bound = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
        layer.Weights.SetRandom(seed, -bound, bound);
    }

    /// <summary>
    /// Finds the index of the maximum value in an array (ArgMax)
    /// </summary>
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

    /// <summary>
    /// Builds and trains a CNN for MNIST digit recognition
    /// Architecture: 28x28x1 -> Conv -> ReLU -> Pool -> Flatten -> Dense -> SoftMax
    /// Uses mini-batch training for stable convergence
    /// </summary>
    private static (GeneralFeedForwardANN, List<Layer>) TrainCNN(
        int epochs, 
        float learningRate = 0.01f,
        int batchSize = 64)
    {
        MNISTTrainingSet trainingSet = new MNISTTrainingSet();
        
        // Get training pairs in 2D format (required for CNN)
        var trainingPairs = trainingSet.BuildNewRandomizedTrainingList(do2DImage: true);
        
        // Input: 28x28 grayscale images (MNIST) - 2D format for CNN
        var inputShape = new InputOutputShape(28, 28, 1, 1);
        
        // Layer 1: Convolution - 5 kernels of size 4x4, stride 1
        // Output: 25x25x5 (because (28-4)/1 + 1 = 25)
        var conv1 = new ConvolutionLayer(inputShape, 
            kernelCount: 5, 
            kernelSquareDimension: 4, 
            stride: 1);
        
        // Layer 2: ReLU activation - preserves spatial dimensions
        var relu1 = new ReLUActivaction(conv1.OutputShape);
        
        // Layer 3: Max Pooling - 2x2 kernels, stride 2
        // Output: 12x12x5 (because 25/2 = 12 with stride 2)
        var pool1 = new PoolingLayer(relu1.OutputShape, 
            stride: 2, 
            kernelCount: 5, 
            kernelSquareDimension: 2, 
            kernelDepth: 1);
        
        // Layer 4: Flatten - convert 3D tensor to 1D vector
        // Output: 12*12*5 = 720 nodes
        var flatten = new FlattenLayer(pool1.OutputShape, nodeCount: 1);
        
        // Layer 5: Dense (fully connected) - 720 -> 10
        var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 10);
        
        // Layer 6: SoftMax - convert logits to probabilities
        var softmax = new SoftMax(dense.OutputShape, nodeCount: 10);
        
        // Build layer stack
        List<Layer> layers = new List<Layer>()
        {
            conv1,
            relu1,
            pool1,
            flatten,
            dense,
            softmax
        };
        
        // Apply proper Xavier initialization to weighted layers
        // This is critical for proper convergence!
        // Conv layer: 4x4 kernel, 1 input channel, 5 output channels
        float convBound = (float)Math.Sqrt(6.0 / (16 + 5));  // fan_in=4*4=16, fan_out=5
        for (int k = 0; k < 5; k++)
        {
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    conv1.Kernels[k, 0][r, c] = (float)(new Random(41 + k).NextDouble() * 2 * convBound - convBound);
            conv1.Biases[k][0, 0] = 0;
        }
        XavierInitialize(dense, 720, 10, seed: 42);
        
        // Create the network with Categorical Cross Entropy loss
        var network = new GeneralFeedForwardANN(
            layers,
            trainingRate: learningRate,
            inputDim: 28 * 28,
            outputDim: 10,
            new CategoricalCrossEntropy());
        
        // Train the network using ConvolutionRenderContext with mini-batch gradient descent
        // This preserves 2D spatial dimensions for proper CNN operations
        Console.WriteLine($"Starting training for {epochs} epochs with {trainingPairs.Count} samples...");
        Console.WriteLine($"Learning rate: {learningRate}, Batch size: {batchSize}");
        Console.WriteLine("Architecture: Conv(4x4,5) -> ReLU -> Pool(2x2) -> Flatten -> Dense(720->10) -> SoftMax");
        
        // Create render context that properly handles 2D images for CNN training
        var renderContext = new ConvolutionRenderContext(network, batchSize, trainingSet);
        
        // Train for specified number of epochs
        // BatchTrain will be called automatically for each epoch
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            ConvolutionRenderContext.BatchTrain(renderContext, epoch);
            
            // Calculate and display epoch statistics
            // Use a sample from the training set to estimate loss/accuracy
            float epochLoss = 0;
            int correct = 0;
            int count = 0;
            int checkInterval = trainingPairs.Count / 10;  // Check 10 samples per epoch
            
            for (int i = 0; i < trainingPairs.Count; i += checkInterval)
            {
                var pair = trainingPairs[i];
                var output = renderContext.FeedForward(pair.Input);
                var predicted = output.ToColumnVector();
                if (predicted != null)
                {
                    epochLoss += network.GetTotallLoss(pair, predicted);
                    if (ArgMax(predicted.Column) == ArgMax(pair.Output.ToColumnVector()!.Column))
                        correct++;
                    count++;
                }
            }
            
            float avgLoss = epochLoss / count;
            float accuracy = (float)correct / count;
            int batchesPerEpoch = trainingPairs.Count / batchSize;
            Console.WriteLine($"Epoch {epoch,2}: Loss={avgLoss:F4}, Accuracy={accuracy:P2} ({correct}/{count}), Batches={batchesPerEpoch}");
        }
        
        Console.WriteLine("Training complete!");
        
        return (network, layers);
    }
}
