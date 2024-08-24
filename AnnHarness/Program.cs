// See https://aka.ms/new-console-template for more information
using System;
using NeuralNets;
using NumReaderNetwork;

class AnnHarness
{
    static int Main(String[] args)
    {
        int hiddenLayer = 32;
        int outputDim = 10;
        double trainingRate = 0.05;

        MNISTSpecificANN ann = new MNISTSpecificANN(hiddenLayer, outputDim, trainingRate);

        int epochs = 10;
        ann.TrainWithImages(epochs);

        return 0;

    }
}


