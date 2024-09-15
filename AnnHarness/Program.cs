// See https://aka.ms/new-console-template for more information
using MnistReader_ANN;
using MatrixLibrary;
using NeuralNets;

class AnnHarness
{
    static int Main(String[] args)
    {
//        DoMyMNIST();
        DoTorchMNIST();

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

        int epochs = 10000;
        ctx.EpochTrain(epochs);
    }
}


