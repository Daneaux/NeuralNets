// See https://aka.ms/new-console-template for more information
using System;
using NeuralNets;
using NumReaderNetwork;

class AnnHarness
{
    static int Main(String[] args)
    {
        int inputDim = 28 * 28;
        int hiddenLayer = 16;
        int outputDim = 10;

        NumberReaderANN ann = new NumberReaderANN(inputDim, hiddenLayer, outputDim);

        int iterations = 1000;
        ann.TrainWithImages(iterations);

        return 0;

    }
}


