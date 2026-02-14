// See https://aka.ms/new-console-template for more information
using MnistReader_ANN;
using MatrixLibrary;
using NeuralNets;
using NeuralNets.Network;
using TorchSharp.Modules;

class AnnHarness
{
    static int Main(String[] args)
    {
        MatrixFactory.SetDefaultBackend(MatrixBackend.GPU);

        //DoMyMNIST();
        //DoTorchMNIST();
       // DoCNN();
       SimpleMnist();

        return 0;

    }

    // 784 -> 16 (relu) -> 16 (relu) -> 10 (sigmoid)
    private static void SimpleMnist()
    {
        MatrixFactory.SetDefaultBackend(MatrixBackend.GPU);

        MNISTTrainingSet trainingSet = new MNISTTrainingSet();

        // Use explicit input shape (28x28x1 for MNIST)
        var inputShape = new InputOutputShape(28, 28, 1, 1);

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
            trainingRate: 0.05f,
            inputDim: inputShape.Width * inputShape.Height,
            outputDim: 10,
            new CategoricalCrossEntropy());

        // Create render context for training
        var ctx = new RenderContext(network, batchSize: 64, trainingSet);

        // Train the network
        int epochs = 10;
        ctx.EpochTrain(epochs);
    }

    private static void DoTorchMNIST()
    {
        TorchMNIST.MNIST.Run(1, 1000, null, null);
    }

    private static void DoMyMNIST()
    {
        float trainingRate = 0.05f;

        MNISTSpecificANN ann1 = new MNISTSpecificANN(trainingRate, 28 * 28, 10); // shouldn't know this. bub bug todo
        RenderContext ctx = new RenderContext(ann1, 256, new MNISTTrainingSet());

        int epochs = 1000;
        ctx.EpochTrain(epochs);
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


