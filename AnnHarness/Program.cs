// See https://aka.ms/new-console-template for more information
using MnistReader_ANN;
using MatrixLibrary;
using NeuralNets;
using System.Diagnostics;

class AnnHarness
{
    static bool doParallel = true;
    static int Main(String[] args)
    {
        MatrixFactory.SetDefaultBackend(MatrixBackend.AVX);
        //MatrixFactory.SetDefaultBackend(MatrixBackend.GPU);

        //DoTorchMNIST();
        // DoCNN();
        (var network, var ctx) = TrainSimpleMnist(epochs: 30, batchSize: 2000, trainingRate: 0.05f);
        RunNetworkOnMnistTestSet(network, ctx);

        return 0;
    }

    private static void RunNetworkOnMnistTestSet(GeneralFeedForwardANN network, RenderContext ctx)
    {
        MNISTTrainingSet trainingSet = new MNISTTrainingSet();
        var testSet = trainingSet.GetTestPairs();
        int totalSamples = 0;
        int totalCorrectSamples = 0;
        foreach (TrainingPair testPair in testSet)
        {
            // run actual vs expected
            var predicted = ctx.FeedForward(testPair.Input);
            var sm = CategoricalCrossEntropy.SoftMax(predicted); // just so i can see the probability distribution.
            var oneHotPredicted = OneHotEncode(predicted.Column);
            var expected = testPair.Output;

            if(IsSamePrediction(oneHotPredicted, expected.ToColumnVector().Column))
            {
                totalCorrectSamples++;
            }
            totalSamples++;
        }
        Console.WriteLine($"Total Samples: {totalSamples}, Total Correct: {totalCorrectSamples}, Accuracy: {(float)totalCorrectSamples / totalSamples}");
    }

    private static bool IsSamePrediction(float[] a, float[] b)
    {
        Debug.Assert(a.Length == b.Length);
        for(int i=0; i < a.Length; i++)
        {
            if(a[i] != b[i])
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


    // 784 -> 16 (relu) -> 16 (relu) -> 10 (cce)
    private static (GeneralFeedForwardANN, RenderContext) TrainSimpleMnist(int epochs, int batchSize = 64, float trainingRate = 0.05f)
    {
        MNISTTrainingSet trainingSet = new MNISTTrainingSet();

        // Use explicit input shape (28x28x1 for MNIST)
        var inputShape = new InputOutputShape(1, 28*28, 1, 1);

        var linear1 = new WeightedLayer(inputShape, nodeCount: 16);
        var relu1 = new ReLUActivaction();

        var linear2 = new WeightedLayer(linear1.OutputShape, nodeCount: 16);
        var relu2 = new ReLUActivaction();

        var output = new WeightedLayer(linear2.OutputShape, nodeCount: 10);

        List<Layer> layers = new List<Layer>()
        {
            linear1,
            relu1,
            linear2,
            relu2,
            output
        };

        // Create the network
        var network = new GeneralFeedForwardANN(
            layers,
            trainingRate: trainingRate,
            inputDim: inputShape.Width * inputShape.Height,
            outputDim: 10,
            new CategoricalCrossEntropy());

        // Create render context for training
        var ctx = new RenderContext(network, batchSize: batchSize, trainingSet);

        // Train the network
        ctx.EpochTrain(epochs, doParallel);

        return (network, ctx);
    }

    private static void DoTorchMNIST()
    {
        TorchMNIST.MNIST.Run(1, 1000, null, null);
    }

    private static void DoCNN()
    {
        MatrixFactory.SetDefaultBackend(MatrixBackend.GPU);

        MNISTTrainingSet trainingSet = new MNISTTrainingSet();
        
        // Use explicit input shape (28x28x1 for MNIST)
        var inputShape = new InputOutputShape(28, 28, 1, 1);
        
        // Build CNN architecture
        var conv1 = new ConvolutionLayer(inputShape, kernelCount: 5, kernelSquareDimension: 4, stride: 1);
        var relu1 = new ReLUActivaction();
        var pool1 = new PoolingLayer(conv1.OutputShape, stride: 2, kernelCount: 5, kernelSquareDimension: 2, kernelDepth: 1);
        var flatten = new FlattenLayer(pool1.OutputShape, nodeCount: 1);
        var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 10);
        var sigmoid = new SigmoidActivation();
        
        List<Layer> layers = new List<Layer>()
        {
            conv1,
            relu1,
            pool1,
            flatten,
            dense,
            sigmoid
        };
        
        // Create the network
        var network = new GeneralFeedForwardANN(
            layers,
            trainingRate: 0.01f,
            inputDim: 28 * 28,
            outputDim: 10,
            new SquaredLoss());
        
        // Create render context for training
        var ctx = new RenderContext(network, batchSize: 64, trainingSet);
        
        // Train the network
        int epochs = 10;
        ctx.EpochTrain(epochs);
    }
}


