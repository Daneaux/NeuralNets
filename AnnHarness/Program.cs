// See https://aka.ms/new-console-template for more information
using MnistReader_ANN;
using MatrixLibrary;
using NeuralNets;
using NeuralNets.Network;

class AnnHarness
{
    static int Main(String[] args)
    {
        MatrixFactory.SetDefaultBackend(MatrixBackend.GPU);

        //DoMyMNIST();
        //DoTorchMNIST();
        DoCNN();

        return 0;

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
        var conv1 = new ConvolutionLayer(trainingSet.OutputShape, 5, 4, 1);
        var p1 = new PoolingLayer(conv1.OutputShape, 2, 5, 2, 1);
        var p2 = new FlattenLayer(p1.OutputShape, 1);
        List<Layer> layers = new List<Layer>()
        {
            conv1,
            new ReLUActivaction(),
            p1,
            p2,
            //new NormalizationLayer(p1.OutputShape, 1),
            new WeightedLayer(p2.OutputShape, 10, p2.OutputShape.TotalFlattenedSize),
            new SigmoidActivation()
        };
        ConvolutionNN nn = new ConvolutionNN(layers, 0.05f, 1, 1, new SquaredLoss());
        ConvolutionRenderContext crn = new ConvolutionRenderContext(nn, 256, trainingSet);

        List<TrainingPair> tps = trainingSet.BuildNewRandomizedTrainingList();

        GeneralFeedForwardANN gann = new GeneralFeedForwardANN(layers, 0.05f, 784, 10, new SquaredLoss());
        RenderContext ctx = new RenderContext(gann, 64, new MNISTTrainingSet());

        int epochs = 1000;
        ctx.EpochTrain(epochs);


        //Tensor t = crn.FeedForward(tps[0].Input);
        
        
    }
}


