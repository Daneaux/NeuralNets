using NeuralNets;

namespace NeuralNetsTests.ANNTests
{
    [TestClass]
    public class ExampleAnn
    {

        public class SineWaveTrainingSet : ITrainingSet
        {
            public double Increment { get; }

            public SineWaveTrainingSet(int numSamples, double increment)
            {
                this.NumberOfSamples = numSamples;
                this.Increment = increment;
            }

            public int InputDimension => 1;

            public int OutputDimension => 1;

            public int NumberOfSamples { get; }

            public int NumberOfLabels => 0;

            public List<TrainingPair> TrainingList { get; private set; }

            public List<TrainingPair> BuildNewRandomizedTrainingList()
            {
                Random rnd = new Random();
                List<TrainingPair> trainingPairs = new List<TrainingPair>();
                double x = 0.0;
                for (int i = 0; i < NumberOfSamples; i++)
                {
                    x = rnd.NextDouble() * 2.0 * System.Math.PI;
                    double sinx = System.Math.Sin(x);
                    trainingPairs.Add(new TrainingPair(new ColumnVector([x]), new ColumnVector([sinx])));
                }
                this.TrainingList = trainingPairs;
                return trainingPairs;
            }
        }
        [TestMethod]
        public void SineWaveTest()
        {
            int inputDim = 1;
            int hiddenLayerDim = 4;
            int outputDim = 1;
            double trainingRate = 0.1;
            int batchSize = 32;
            ILossFunction lossFunction = new SquaredLoss();

            WeightedLayer hiddenLayer = new WeightedLayer(hiddenLayerDim, new SigmoidActivation(), inputDim);
            WeightedLayer outputLayer = new WeightedLayer(outputDim, new SigmoidActivation(), hiddenLayerDim);

            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(
                new List<WeightedLayer> { hiddenLayer, outputLayer },
                trainingRate,
                batchSize,
                lossFunction,
                new SineWaveTrainingSet(1000, 0.05));

            ann.EpochTrain(10);
            Console.WriteLine($"tOTAL Loss = {ann.GetTotallLoss}\n");
        }

        public class MazurTrainingSet : ITrainingSet
        {
            public MazurTrainingSet(TrainingPair tp)
            {
                this.TrainingPair = tp;
            }

            public int InputDimension => 2;

            public int OutputDimension => 2;

            public int NumberOfSamples => 1;

            public int NumberOfLabels => 0;

            public TrainingPair TrainingPair { get; }

            public List<TrainingPair> TrainingList { get; private set; }

            public List<TrainingPair> BuildNewRandomizedTrainingList()
            {
                return null;
            }
        }

        [TestMethod]
        public void MattMazurExample()
        {
            double trainingRate = 0.5;
            int inputDim = 2;
            int batchSize = 1;

            //TrainingData trainingData = new TrainingData(inputDim, outputDim, new List<TrainingPair>());
            TrainingPair tp = new(new ColumnVector([0.05, 0.10]), new ColumnVector([0.01, 0.99]));

           //trainingData.TrainingPairs.Add(tp);

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
                new List<WeightedLayer> { hiddenLayer, outputLayer }, 
                trainingRate, 
                batchSize,
                null,
                new MazurTrainingSet(tp));

            //ann.TrainingData = trainingData;

            double totLoss = 0;

            ColumnVector finalOutput = ann.FeedForward(tp.Input);

            Assert.AreEqual(0.593269992, hiddenLayer.LastActivationOutput[0], 0.000001);
            Assert.AreEqual(0.596884378, hiddenLayer.LastActivationOutput[1], 0.000001);

            Assert.AreEqual(0.75136507, outputLayer.LastActivationOutput[0], 0.000001);
            Assert.AreEqual(0.772928465, outputLayer.LastActivationOutput[1], 0.000001);


            totLoss = ann.GetTotallLoss(tp, finalOutput);
            Assert.AreEqual(0.75136507, finalOutput[0], 0.00001);
            Assert.AreEqual(0.772928465, finalOutput[1], 0.00001);

            ColumnVector lossVector = ann.GetLossVector(tp, finalOutput);
            Assert.AreEqual(0.274811083, lossVector[0], 0.00001);
            Assert.AreEqual(0.023560026, lossVector[1], 0.00001);

            Assert.AreEqual(0.298371109, totLoss, 0.00001);

            //
            // Do backprop
            //

            //ann.BackProp_2layer(tp, finalOutput);
            ann.BackProp(tp, finalOutput);
            outputLayer.ScaleWeightAndBiasesGradient(trainingRate);
            hiddenLayer.ScaleWeightAndBiasesGradient(trainingRate);
            ann.UpdateWeightsAndBiases();


            Assert.AreEqual(outputLayer.Weights[0, 0], 0.35891648, 0.00001);
            Assert.AreEqual(outputLayer.Weights[0, 1], 0.408666186, 0.00001);
            Assert.AreEqual(outputLayer.Weights[1, 0], 0.51130270, 0.00001);
            Assert.AreEqual(outputLayer.Weights[1, 1], 0.561370121, 0.00001);

            Assert.AreEqual(hiddenLayer.Weights[0, 0], 0.149780716, 0.00001);
            Assert.AreEqual(hiddenLayer.Weights[0, 1], 0.19956143, 0.00001);
            Assert.AreEqual(hiddenLayer.Weights[1, 0], 0.24975114, 0.00001);
            Assert.AreEqual(hiddenLayer.Weights[1, 1], 0.29950229, 0.00001);

            // 
            // hand rolled feed forward!
            //
            double z1 = hiddenLayer.Weights[0, 0] * tp.Input[0] + hiddenLayer.Weights[0, 1] * tp.Input[1] + hiddenLayer.Biases[0];
            double z2 = hiddenLayer.Weights[1, 0] * tp.Input[0] + hiddenLayer.Weights[1, 1] * tp.Input[1] + hiddenLayer.Biases[1];

            ColumnVector a = hiddenLayer.ActivationFunction.Activate(new ColumnVector([z1, z2]));

            double oz1 = outputLayer.Weights[0, 0] * a[0] + outputLayer.Weights[0, 1] * a[1] + outputLayer.Biases[0];
            double oz2 = outputLayer.Weights[1, 0] * a[0] + outputLayer.Weights[1, 1] * a[1] + outputLayer.Biases[1];

            ColumnVector oa = outputLayer.ActivationFunction.Activate(new ColumnVector([oz1, oz2]));

            // truth - predicted
            double error1 = 0.5 * (tp.Output[0] - oa[0]) * (tp.Output[0] - oa[0]);
            double error2 = 0.5 * (tp.Output[1] - oa[1]) * (tp.Output[1] - oa[1]);
            double totalErrorPass2 = error1 + error2;


            //
            // feed forward and test total loss
            //
            finalOutput = ann.FeedForward(tp.Input);
            totLoss = ann.GetTotallLoss(tp, finalOutput);
            Assert.AreEqual(0.28047, totLoss, 0.001);


            Console.WriteLine("done error is " + totLoss);

            totLoss = 0;
            for (int i = 0; i < 10000; i++)
            {
                ann.BatchTrain();
                totLoss = ann.GetTotallLoss(tp, outputLayer.LastActivationOutput);
            }

            //totLoss = ann.GetTotallLoss(tp, ann.FeedForward(tp.Input));
            Console.WriteLine("done error is " + totLoss);
            Assert.AreEqual(totLoss, 1.5081705298794045E-07, 0.00000001);
        }
    }
}
