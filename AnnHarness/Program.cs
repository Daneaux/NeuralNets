// See https://aka.ms/new-console-template for more information
using MnistReader_ANN;
using MatrixLibrary;
using NeuralNets;
using NeuralNets.Network;

class AnnHarness
{
    static int Main(String[] args)
    {
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
        MNISTTrainingSet trainingSet = new MNISTTrainingSet();
        ConvolutionLayer conv1 = new ConvolutionLayer(trainingSet.OutputShape, 4, 4, new ReLUActivaction(), 1);
        PoolingLayer p1 = new PoolingLayer(conv1.OutputShape, 2, 5, 2, 1);
        WeightedLayer w1 = new WeightedLayer(p1.OutputShape, 10, new SigmoidActivation(), p1.FlatOutputSize);
        ConvolutionNN nn = new ConvolutionNN(new List<Layer> { conv1, p1, w1 }, 0.05f, 1, 1, new SquaredLoss());
        ConvolutionRenderContext crn = new ConvolutionRenderContext(nn, 256, trainingSet);

        List<TrainingPair> tps = trainingSet.BuildNewRandomizedTrainingList();

        Tensor t = crn.FeedForward(tps[0].Input);

        
    }
}


