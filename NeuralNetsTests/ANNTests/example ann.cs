using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NeuralNets;
using NumReaderNetwork;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace NeuralNetsTests.ANNTests
{
    [TestClass]
    public class ExampleAnn
    {
        [TestMethod]
        public void SineWaveTest()
        {
            int inputDim = 1;
            int hiddenLayerDim = 4;
            int outputDim = 1;
            double trainingRate = 0.1;
            ILossFunction lossFunction = new SquaredLoss();


            TrainingData trainingData = new TrainingData(inputDim, outputDim, new List<TrainingPair>());

            //int j = 0;
            for(double i=0; i<1000.0; i+=0.01)
            {
                double x = System.Math.Sin(i);
                trainingData.TrainingPairs.Add(new(new ColumnVector([i]), new ColumnVector([x])));
            }

            WeightedLayer hiddenLayer = new WeightedLayer(hiddenLayerDim, new SigmoidActivation(), inputDim);
            WeightedLayer outputLayer = new WeightedLayer(outputDim, new SigmoidActivation(), hiddenLayerDim);

            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(inputDim, new List<WeightedLayer> { hiddenLayer, outputLayer }, trainingRate, lossFunction);

            for (int x = 0; x < 2; x++)
            {
                double error = 0;
                for (int i = 0; i < 100; i++)
                {
                    error = 0;
                    foreach (TrainingPair tp in trainingData.TrainingPairs)
                    {
                        ColumnVector predicted = ann.FeedForward(tp.Input);
                        error += ann.GetAveragelLoss(tp, predicted);
                        ann.BackProp(tp, predicted);
                    }

                    double averageError = error / (double)trainingData.TrainingPairs.Count;
                    Console.WriteLine($"{x}: Average Loss = {averageError}\n");
                }

            }
        }


        [TestMethod]
        public void MattMazurExample()
        {
            double trainingRate = 0.5;
            int inputDim = 2;
            int hiddenLayerDim = 2;
            int outputDim = 2;

            TrainingData trainingData = new TrainingData(inputDim, outputDim, new List<TrainingPair>());
            TrainingPair tp = new(new ColumnVector([0.05, 0.10]), new ColumnVector([0.01, 0.99]));

            trainingData.TrainingPairs.Add(tp);

            Matrix w1 = new Matrix(new double[,] {
                { 0.15, 0.2 },
                { 0.25, 0.30 }
            });

            Matrix w2 = new Matrix(new double[,] {
                { 0.40, 0.45 },
                { 0.50, 0.55 }
            });

            ColumnVector b1 = new ColumnVector(new double[] { 0.35, 0.35 });
            ColumnVector b2 = new ColumnVector(new double[] { 0.60, 0.60 });


            WeightedLayer hiddenLayer = new WeightedLayer(2, new SigmoidActivation(), 2, w1, b1);
            WeightedLayer outputLayer = new WeightedLayer(2, new SigmoidActivation(), 2, w2, b2);


            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(
                inputDim, 
                new List<WeightedLayer> { hiddenLayer, outputLayer }, 
                trainingRate, null);

            double totLoss = 0 ;
            for (int i = 0; i < 10000; i++)
            {
                ColumnVector finalOutput = ann.FeedForward(trainingData.TrainingPairs[0].Input);
                totLoss = ann.GetTotallLoss(tp, finalOutput);
                ann.BackProp(tp, finalOutput);
                //ann.BackProp_2layer(tp, finalOutput);
            }
            Console.WriteLine("done error is " + totLoss);
            Assert.AreEqual(totLoss, 0.0000024483375576262793, 0.00000001);

        }
    }
}
