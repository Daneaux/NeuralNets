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
        double trainingRate = 0.05;

        MNISTSpecificANN ann = new MNISTSpecificANN(trainingRate, 28 * 28, 10); // shouldn't know this. bub bug todo
        RenderContext ctx = new RenderContext(ann, 128, new MNISTTrainingSet());

        int epochs = 10;
        ctx.EpochTrain(epochs);

        return 0;

    }
}


