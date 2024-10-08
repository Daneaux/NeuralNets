﻿using MatrixLibrary;
using NeuralNets;
using SharpTorch;
using System.Diagnostics;

namespace NeuralNetsTests.ANNTests
{
    [TestClass]
    public class ExampleAnn
    {

        public class SineWaveTrainingSet : ITrainingSet
        {
            public float Increment { get; }

            private readonly int seed;

            public SineWaveTrainingSet(int numSamples, float increment, int seed)
            {
                this.NumberOfSamples = numSamples;
                this.Increment = increment;
                this.seed = seed;
            }

            public int InputDimension => 1;

            public int NumClasses => 1;

            public int NumberOfSamples { get; }

            public int NumberOfLabels => 0;

            public List<TrainingPair> TrainingList { get; private set; }

            public int Width => 1;

            public int Height => 1;

            public int Depth => 1;

            public InputOutputShape OutputShape => new InputOutputShape(1, 1, 1, 1);

            public List<TrainingPair> BuildNewRandomizedTrainingList()
            {
                float range = 10F;
                Random rnd = new Random(this.seed);
                List<TrainingPair> trainingPairs = new List<TrainingPair>();
                float x = 0;
                for (int i = 0; i < NumberOfSamples; i++)
                {
                    x = (float)(rnd.NextDouble() * 2 * System.Math.PI) * range;
                    float sinx = (float)System.Math.Sin(x);
                    trainingPairs.Add(new TrainingPair(new AvxColumnVector([x]), new AvxColumnVector([sinx])));
                }
                this.TrainingList = trainingPairs;
                return trainingPairs;
            }
        }
        [TestMethod]
        public void SineWaveTest()
        {
            int inputDim = 1;
            int hiddenLayerDim = 16;
            int outputDim = 1;
            float trainingRate = 0.025F;
            int batchSize = 32;
            ILossFunction lossFunction = new SquaredLoss();

            WeightedLayer hiddenLayer1 = new WeightedLayer(new InputOutputShape(1, 1, 1, 1), hiddenLayerDim, new SigmoidActivation());
            WeightedLayer hiddenLayer2 = new WeightedLayer(hiddenLayer1.OutputShape, hiddenLayerDim, new SigmoidActivation());
            WeightedLayer outputLayer = new WeightedLayer(hiddenLayer2.OutputShape, outputDim, new SigmoidActivation());

            int trainingSamples = 40000;
            ITrainingSet ts = new SineWaveTrainingSet(trainingSamples, 0.001F, 09870987);
            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(
                new List<Layer> { hiddenLayer1, hiddenLayer2, outputLayer },
                trainingRate,
                inputDim,
                outputDim,
                lossFunction);
            RenderContext ctx = new RenderContext(ann, 1, ts);

            int numEpochs = 100;
            ctx.EpochTrain(numEpochs);


            ITrainingSet ts2 = new SineWaveTrainingSet(1000, 0.05F, 1987987234);
            float averageLoss = 0f;
            int testSamples = 1000;
            for (int i = 0; i < testSamples; i++)
            {
                RenderContext lossCtx = new RenderContext(ann, 1, ts2);
                AvxColumnVector prediction = lossCtx.FeedForward(ts.TrainingList[i].Input);
                averageLoss += ann.GetAveragelLoss(ts.TrainingList[i], prediction);
            }

            averageLoss /= testSamples;
            Debug.WriteLine($"{averageLoss} after {testSamples} test samples and {trainingSamples} and {numEpochs} epochs\n");            
        }

        public class MazurTrainingSet : ITrainingSet
        {
            public MazurTrainingSet(TrainingPair tp)
            {
                this.TrainingPair = tp;
            }

            public int InputDimension => 2;

            public int NumClasses => 2;

            public int NumberOfSamples => 1;

            public int NumberOfLabels => 0;

            public TrainingPair TrainingPair { get; }

            public List<TrainingPair> TrainingList { get; private set; }

            public int Width => 1;

            public int Height => 2;

            public int Depth => 1;

            public InputOutputShape OutputShape => new InputOutputShape(1, 2, 1, 1);

            public List<TrainingPair> BuildNewRandomizedTrainingList()
            {
                return new List<TrainingPair> { new(new AvxColumnVector([0.05F, 0.10F]), new AvxColumnVector([0.01F, 0.99F])) };
            }
        }

        [TestMethod]
        public void MattMazurExample()
        {
            //SimpleNN snn = new SimpleNN(;)
            SimpleNN.DoTrainOnePass();

            float trainingRate = 0.5F;
            int inputDim = 2;
            int batchSize = 1;
            int outputDim = 2;

            TrainingPair tp = new(new AvxColumnVector([0.05F, 0.10F]), new AvxColumnVector([0.01F, 0.99F]));

            AvxMatrix w1 = new AvxMatrix(new float[,] {
                { 0.15F, 0.2F },
                { 0.25F, 0.30F }
            });

            AvxMatrix w2 = new AvxMatrix(new float[,] {
                { 0.40f, 0.45f },
                { 0.50f, 0.55f }
            });

            AvxColumnVector b1 = new AvxColumnVector(new float[] { 0.35f, 0.35f });
            AvxColumnVector b2 = new AvxColumnVector(new float[] { 0.60f, 0.60f });

            InputOutputShape shape = new InputOutputShape(1, 2, 1, 1);
            WeightedLayer hiddenLayer = new WeightedLayer(shape, 2, new SigmoidActivation(), w1, b1);
            WeightedLayer outputLayer = new WeightedLayer(shape, 2, new SigmoidActivation(), w2, b2);

            MazurTrainingSet ts = new MazurTrainingSet(tp);

            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(
                new List<Layer> { hiddenLayer, outputLayer },
                trainingRate,
                inputDim,
                outputDim,
                new SquaredLoss());                

            RenderContext ctx = new RenderContext(ann, batchSize, ts);

            float totLoss = 0;

            AvxColumnVector finalOutput = ctx.FeedForward(tp.Input);

            AvxColumnVector hiddenLayerActivation = ctx.ActivationContext[0];
            AvxColumnVector outputLayerActivation = ctx.ActivationContext[1];

            Assert.AreEqual(0.593269992, hiddenLayerActivation[0], 0.000001);
            Assert.AreEqual(0.596884378, hiddenLayerActivation[1], 0.000001);

            Assert.AreEqual(0.75136507, outputLayerActivation[0], 0.000001);
            Assert.AreEqual(0.772928465, outputLayerActivation[1], 0.000001);


            totLoss = ann.GetTotallLoss(tp, finalOutput);
            Assert.AreEqual(0.75136507, finalOutput[0], 0.00001);
            Assert.AreEqual(0.772928465, finalOutput[1], 0.00001);

            AvxColumnVector lossVector = ann.GetLossVector(tp, finalOutput);
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
            float z1 = hiddenLayer.Weights[0, 0] * tp.Input.ToAvxColumnVector()[0] + hiddenLayer.Weights[0, 1] * tp.Input.ToAvxColumnVector()[1] + hiddenLayer.Biases[0];
            float z2 = hiddenLayer.Weights[1, 0] * tp.Input.ToAvxColumnVector()[0] + hiddenLayer.Weights[1, 1] * tp.Input.ToAvxColumnVector()[1] + hiddenLayer.Biases[1];

            AvxColumnVector a = hiddenLayer.Activate(new AvxColumnVector([z1, z2]));

            float oz1 = outputLayer.Weights[0, 0] * a[0] + outputLayer.Weights[0, 1] * a[1] + outputLayer.Biases[0];
            float oz2 = outputLayer.Weights[1, 0] * a[0] + outputLayer.Weights[1, 1] * a[1] + outputLayer.Biases[1];

            AvxColumnVector oa = outputLayer.Activate(new AvxColumnVector([oz1, oz2]));

            // truth - predicted
            float error1 = 0.5f * (tp.Output.ToAvxColumnVector()[0] - oa[0]) * (tp.Output.ToAvxColumnVector()[0] - oa[0]);
            float error2 = 0.5f * (tp.Output.ToAvxColumnVector()[1] - oa[1]) * (tp.Output.ToAvxColumnVector()[1] - oa[1]);
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
                AvxColumnVector pout = ctx4.FeedForward(tp.Input);
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

            TrainingPair tp = new(new AvxColumnVector([0.05f, 0.10f]), new AvxColumnVector([0.01f, 0.99f]));

            AvxMatrix w1 = new AvxMatrix(new float[,] {
                { 0.15f, 0.2f },
                { 0.25f, 0.30f }
            });

            AvxMatrix w2 = new AvxMatrix(new float[,] {
                { 0.40f, 0.45f },
                { 0.50f, 0.55f }
            });

            AvxColumnVector b1 = new AvxColumnVector(new float[] { 0.35f, 0.35f });
            AvxColumnVector b2 = new AvxColumnVector(new float[] { 0.60f, 0.60f });


            InputOutputShape shape = new InputOutputShape(1, 2, 1, 1);
            WeightedLayer hiddenLayer = new WeightedLayer(shape, 2, new SigmoidActivation(), w1, b1);
            WeightedLayer outputLayer = new WeightedLayer(shape, 2, new SigmoidActivation(), w2, b2);

            MazurTrainingSet ts = new MazurTrainingSet(tp);

            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(
                new List<Layer> { hiddenLayer, outputLayer },
                trainingRate,
                inputDim,
                outputDim,
                new SquaredLoss());

            RenderContext ctx = new RenderContext(ann, batchSize, ts);

            float totLoss = 0;

            AvxColumnVector finalOutput = ctx.FeedForward(tp.Input);

            AvxColumnVector hiddenLayerActivation = ctx.ActivationContext[0];
            AvxColumnVector outputLayerActivation = ctx.ActivationContext[1];

            Assert.AreEqual(0.593269992, hiddenLayerActivation[0], 0.000001);
            Assert.AreEqual(0.596884378, hiddenLayerActivation[1], 0.000001);

            Assert.AreEqual(0.75136507, outputLayerActivation[0], 0.000001);
            Assert.AreEqual(0.772928465, outputLayerActivation[1], 0.000001);


            totLoss = ann.GetTotallLoss(tp, finalOutput);
            Assert.AreEqual(0.75136507, finalOutput[0], 0.00001);
            Assert.AreEqual(0.772928465, finalOutput[1], 0.00001);

            AvxColumnVector lossVector = ann.GetLossVector(tp, finalOutput);
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
                AvxColumnVector pout = ctx4.FeedForward(tp.Input);
                totLoss = ctx4.Network.GetTotallLoss(tp, pout);
            }

            Console.WriteLine("done error is " + totLoss);
            Assert.AreEqual(totLoss, 2.4483375576262793E-06, 0.00000001);
        }

        [TestMethod]
        public void MattMazurExample_Softmax_comparePyTorch()
        {
            float trainingRate = 0.5f;
            int inputDim = 2;
            int batchSize = 1;
            int outputDim = 2;

            TrainingPair tp = new(new AvxColumnVector([0.05f, 0.10f]), new AvxColumnVector([0.01f, 0.99f]));

            AvxMatrix w1 = new AvxMatrix(new float[,] {
                { 0.15f, 0.2f },
                { 0.25f, 0.30f }
            });

            AvxMatrix w2 = new AvxMatrix(new float[,] {
                { 0.40f, 0.45f },
                { 0.50f, 0.55f }
            });

            AvxColumnVector b1 = new AvxColumnVector(new float[] { 0.35f, 0.35f });
            AvxColumnVector b2 = new AvxColumnVector(new float[] { 0.60f, 0.60f });

            InputOutputShape shape = new InputOutputShape(1, 2, 1, 1);
            WeightedLayer hiddenLayer = new WeightedLayer(shape, 2, new SigmoidActivation(), w1, b1);
            WeightedLayer outputLayer = new WeightedLayer(shape, 2, new SigmoidActivation(), w2, b2);

            MazurTrainingSet ts = new MazurTrainingSet(tp);

            GeneralFeedForwardANN ann = new GeneralFeedForwardANN(
                new List<Layer> { hiddenLayer, outputLayer },
                trainingRate,
                inputDim,
                outputDim,
                new SquaredLoss());

            RenderContext ctx = new RenderContext(ann, batchSize, ts);

            float totLoss = 0;

            AvxColumnVector finalOutput = ctx.FeedForward(tp.Input);

            AvxColumnVector hiddenLayerActivation = ctx.ActivationContext[0];
            AvxColumnVector outputLayerActivation = ctx.ActivationContext[1];

            Assert.AreEqual(0.593269992, hiddenLayerActivation[0], 0.000001);
            Assert.AreEqual(0.596884378, hiddenLayerActivation[1], 0.000001);

            Assert.AreEqual(0.75136507, outputLayerActivation[0], 0.000001);
            Assert.AreEqual(0.772928465, outputLayerActivation[1], 0.000001);


            totLoss = ann.GetTotallLoss(tp, finalOutput);
            Assert.AreEqual(0.75136507, finalOutput[0], 0.00001);
            Assert.AreEqual(0.772928465, finalOutput[1], 0.00001);

            AvxColumnVector lossVector = ann.GetLossVector(tp, finalOutput);
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
                AvxColumnVector pout = ctx4.FeedForward(tp.Input);
                totLoss = ctx4.Network.GetTotallLoss(tp, pout);
            }

            Console.WriteLine("done error is " + totLoss);
            Assert.AreEqual(totLoss, 2.4483375576262793E-06, 0.00000001);
        }

    }
}
