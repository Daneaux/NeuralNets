using NeuralNets;
using NeuralNets.Network;

namespace NeuralNetsTests.ANNTests
{
    [TestClass]
    public class ExampleAnn
    {

        public class SineWaveTrainingSet : ITrainingSet
        {
            public float Increment { get; }

            public SineWaveTrainingSet(int numSamples, float increment)
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
                float x = 0;
                for (int i = 0; i < NumberOfSamples; i++)
                {
                    x = (float)(rnd.NextDouble() * 2 * System.Math.PI);
                    float sinx = (float)System.Math.Sin(x);
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
            float trainingRate = 0.1F;
            int batchSize = 32;
            ILossFunction lossFunction = new SquaredLoss();

            WeightedLayer hiddenLayer = new WeightedLayer(hiddenLayerDim, new SigmoidActivation(), inputDim);
            WeightedLayer outputLayer = new WeightedLayer(outputDim, new SigmoidActivation(), hiddenLayerDim);

            ITrainingSet ts = new SineWaveTrainingSet(1000, 0.05F);
            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(
                new List<WeightedLayer> { hiddenLayer, outputLayer },
                trainingRate,
                inputDim,
                outputDim,
                lossFunction);
            RenderContext ctx = new RenderContext(ann, 1, ts);
            ctx.EpochTrain(10);
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
                return new List<TrainingPair> { new(new ColumnVector([0.05F, 0.10F]), new ColumnVector([0.01F, 0.99F])) };
            }
        }

        [TestMethod]
        public void MattMazurExample()
        {

            float trainingRate = 0.5F;
            int inputDim = 2;
            int batchSize = 1;
            int outputDim = 2;

            TrainingPair tp = new(new ColumnVector([0.05F, 0.10F]), new ColumnVector([0.01F, 0.99F]));

            Matrix w1 = new Matrix(new float[,] {
                { 0.15F, 0.2F },
                { 0.25F, 0.30F }
            });

            Matrix w2 = new Matrix(new float[,] {
                { 0.40f, 0.45f },
                { 0.50f, 0.55f }
            });

            ColumnVector b1 = new ColumnVector(new float[] { 0.35f, 0.35f });
            ColumnVector b2 = new ColumnVector(new float[] { 0.60f, 0.60f });


            WeightedLayer hiddenLayer = new WeightedLayer(2, new SigmoidActivation(), 2, w1, b1);
            WeightedLayer outputLayer = new WeightedLayer(2, new SigmoidActivation(), 2, w2, b2);

            MazurTrainingSet ts = new MazurTrainingSet(tp);

            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(
                new List<WeightedLayer> { hiddenLayer, outputLayer },
                trainingRate,
                inputDim,
                outputDim,
                new SquaredLoss());                

            RenderContext ctx = new RenderContext(ann, batchSize, ts);

            float totLoss = 0;

            ColumnVector finalOutput = ctx.FeedForward(tp.Input);

            ColumnVector hiddenLayerActivation = ctx.ActivationContext[0];
            ColumnVector outputLayerActivation = ctx.ActivationContext[1];

            Assert.AreEqual(0.593269992, hiddenLayerActivation[0], 0.000001);
            Assert.AreEqual(0.596884378, hiddenLayerActivation[1], 0.000001);

            Assert.AreEqual(0.75136507, outputLayerActivation[0], 0.000001);
            Assert.AreEqual(0.772928465, outputLayerActivation[1], 0.000001);


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
            ctx.BackProp(tp, finalOutput);
            ctx.ScaleAndUpdateWeightsBiasesHelper(0);
            ctx.ScaleAndUpdateWeightsBiasesHelper(1);

            Assert.AreEqual(outputLayer.Weights[0, 0], 0.35891648, 0.00001);
            Assert.AreEqual(outputLayer.Weights[0, 1], 0.408666186, 0.00001);
            Assert.AreEqual(outputLayer.Weights[1, 0], 0.51130270, 0.00001);
            Assert.AreEqual(outputLayer.Weights[1, 1], 0.561370121, 0.00001);

            Assert.AreEqual(hiddenLayer.Weights[0, 0], 0.149780716, 0.00001);
            Assert.AreEqual(hiddenLayer.Weights[0, 1], 0.19956143, 0.00001);
            Assert.AreEqual(hiddenLayer.Weights[1, 0], 0.24975114, 0.00001);
            Assert.AreEqual(hiddenLayer.Weights[1, 1], 0.29950229, 0.00001);

            // 
            // hand rolled feed forward and test!
            //
            float z1 = hiddenLayer.Weights[0, 0] * tp.Input[0] + hiddenLayer.Weights[0, 1] * tp.Input[1] + hiddenLayer.Biases[0];
            float z2 = hiddenLayer.Weights[1, 0] * tp.Input[0] + hiddenLayer.Weights[1, 1] * tp.Input[1] + hiddenLayer.Biases[1];

            ColumnVector a = hiddenLayer.Activate(new ColumnVector([z1, z2]));

            float oz1 = outputLayer.Weights[0, 0] * a[0] + outputLayer.Weights[0, 1] * a[1] + outputLayer.Biases[0];
            float oz2 = outputLayer.Weights[1, 0] * a[0] + outputLayer.Weights[1, 1] * a[1] + outputLayer.Biases[1];

            ColumnVector oa = outputLayer.Activate(new ColumnVector([oz1, oz2]));

            // truth - predicted
            float error1 = 0.5f * (tp.Output[0] - oa[0]) * (tp.Output[0] - oa[0]);
            float error2 = 0.5f * (tp.Output[1] - oa[1]) * (tp.Output[1] - oa[1]);
            float totalErrorPass2 = error1 + error2;
            Assert.AreEqual(0.28047, totalErrorPass2, 0.001);


            //
            // feed forward and test total loss
            //
            RenderContext ctx3 = new RenderContext(ann, 1, ts);
            finalOutput = ctx3.FeedForward(tp.Input);
            totLoss = ann.GetTotallLoss(tp, finalOutput);
            Assert.AreEqual(0.28047, totLoss, 0.001);
            Console.WriteLine("done error is " + totLoss);

            totLoss = 0;
            for (int i = 0; i < 10000; i++)
            {
                RenderContext.BatchTrain(ctx3, 1);
            }

            {
                RenderContext ctx4 = new RenderContext(ann, 1, ts);
                ColumnVector pout = ctx4.FeedForward(tp.Input);
                totLoss = ctx4.Network.GetTotallLoss(tp, pout);
            }

            Console.WriteLine("done error is " + totLoss);
            Assert.AreEqual(totLoss, 2.4483375576262793E-06, 0.00000001);
        }

        [TestMethod]
        public void MattMazurExample2()
        {

            float trainingRate = 0.5f;
            int inputDim = 2;
            int batchSize = 1;
            int outputDim = 2;

            TrainingPair tp = new(new ColumnVector([0.05f, 0.10f]), new ColumnVector([0.01f, 0.99f]));

            Matrix w1 = new Matrix(new float[,] {
                { 0.15f, 0.2f },
                { 0.25f, 0.30f }
            });

            Matrix w2 = new Matrix(new float[,] {
                { 0.40f, 0.45f },
                { 0.50f, 0.55f }
            });

            ColumnVector b1 = new ColumnVector(new float[] { 0.35f, 0.35f });
            ColumnVector b2 = new ColumnVector(new float[] { 0.60f, 0.60f });


            WeightedLayer hiddenLayer = new WeightedLayer(2, new SigmoidActivation(), 2, w1, b1);
            WeightedLayer outputLayer = new WeightedLayer(2, new SigmoidActivation(), 2, w2, b2);

            MazurTrainingSet ts = new MazurTrainingSet(tp);

            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(
                new List<WeightedLayer> { hiddenLayer, outputLayer },
                trainingRate,
                inputDim,
                outputDim,
                new SquaredLoss());

            RenderContext ctx = new RenderContext(ann, batchSize, ts);

            float totLoss = 0;

            ColumnVector finalOutput = ctx.FeedForward(tp.Input);

            ColumnVector hiddenLayerActivation = ctx.ActivationContext[0];
            ColumnVector outputLayerActivation = ctx.ActivationContext[1];

            Assert.AreEqual(0.593269992, hiddenLayerActivation[0], 0.000001);
            Assert.AreEqual(0.596884378, hiddenLayerActivation[1], 0.000001);

            Assert.AreEqual(0.75136507, outputLayerActivation[0], 0.000001);
            Assert.AreEqual(0.772928465, outputLayerActivation[1], 0.000001);


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

            RenderContext parentContext = new RenderContext(ann, 1, ts);
            RenderContext.BatchTrain(parentContext, 1);

            Assert.AreEqual(outputLayer.Weights[0, 0], 0.35891648, 0.00001);
            Assert.AreEqual(outputLayer.Weights[0, 1], 0.408666186, 0.00001);
            Assert.AreEqual(outputLayer.Weights[1, 0], 0.51130270, 0.00001);
            Assert.AreEqual(outputLayer.Weights[1, 1], 0.561370121, 0.00001);

            Assert.AreEqual(hiddenLayer.Weights[0, 0], 0.149780716, 0.00001);
            Assert.AreEqual(hiddenLayer.Weights[0, 1], 0.19956143, 0.00001);
            Assert.AreEqual(hiddenLayer.Weights[1, 0], 0.24975114, 0.00001);
            Assert.AreEqual(hiddenLayer.Weights[1, 1], 0.29950229, 0.00001);


            //
            // feed forward and test total loss
            //
            RenderContext ctx3 = new RenderContext(ann, 1, ts);
            finalOutput = ctx3.FeedForward(tp.Input);
            totLoss = ann.GetTotallLoss(tp, finalOutput);
            Assert.AreEqual(0.28047, totLoss, 0.001);

            Console.WriteLine("done error is " + totLoss);

            totLoss = 0;
            for (int i = 0; i < 10000; i++)
            {
                RenderContext ctx4 = new RenderContext(ann, 1, ts);
                RenderContext.BatchTrain(ctx4, 1);
            }

            {
                RenderContext ctx4 = new RenderContext(ann, 1, ts);
                ColumnVector pout = ctx4.FeedForward(tp.Input);
                totLoss = ctx4.Network.GetTotallLoss(tp, pout);
            }

            Console.WriteLine("done error is " + totLoss);
            Assert.AreEqual(totLoss, 2.4483375576262793E-06, 0.00000001);
        }
    }
}
