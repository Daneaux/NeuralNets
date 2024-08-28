// See https://aka.ms/new-console-template for more information
using System;
using NeuralNets;
using NumReaderNetwork;

class AnnHarness
{
    static int Main(String[] args)
    {
        double trainingRate = 0.05;

        MNISTSpecificANN ann = new MNISTSpecificANN(trainingRate);

        int epochs = 10;
        ann.EpochTrain(epochs);

        return 0;

    }
}


