// See https://aka.ms/new-console-template for more information
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNets;
using NumReaderNetwork;

class AnnHarness
{
    static int Main(String[] args)
    {
        int inputDim = 3;
        int hiddenLayer = 3;
        int outputDim = 3;
        TrainingPair tp = new TrainingPair(new ColumnVector([0.5, 0.3, 0.2]), new ColumnVector([0.8, 0.6, 0.4]));
        TrainingData trainingData = new TrainingData(inputDim, outputDim, new TrainingPair[] { tp });

        NumberReaderANN ann = new NumberReaderANN(inputDim, hiddenLayer, outputDim);

        for (int i = 0; i < 100; i++)
        {
            ColumnVector predicted = ann.FeedForward(tp.Input);
            double error = ann.GetAveragelLoss(tp, predicted);
            ann.BackProp(tp, predicted);
            Console.WriteLine($"{i}: Loss = {error}\n");
        }

        return 0;

    }
}


