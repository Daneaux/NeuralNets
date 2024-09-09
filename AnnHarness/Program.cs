// See https://aka.ms/new-console-template for more information
using System;
using MnistReader_ANN;
using NeuralNets;
using NeuralNets.Network;
using NumReaderNetwork;

class AnnHarness
{
    static int Main(String[] args)
    {
        float trainingRate = 0.05f;

        MNISTSpecificANN ann = new MNISTSpecificANN(trainingRate, 28 * 28, 10); // shouldn't know this. bub bug todo
        RenderContext ctx = new RenderContext(ann, 256, new MNISTTrainingSet());

        int epochs = 10000;
        ctx.EpochTrain(epochs);

        return 0;

    }
}


