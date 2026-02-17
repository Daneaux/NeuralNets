using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace NeuralNets
{
    public class RenderContext
    {
        public int BatchSize { get; }
        public GeneralFeedForwardANN Network { get; }
        public int CurrentThreadID { get; private set; }
        public ColumnVectorBase[] Sigma { get; private set; }
        public ColumnVectorBase[] ActivationContext { get; }
        public ColumnVectorBase[] DerivativeContext { get; }
        public bool DoRandomSamples { get; private set; }
        public virtual ITrainingSet TrainingSet { get; }

        public int InputDim => Network.InputDim;
        public int OutputDim => Network.OutputDim;
        public int LayerCount => Network.LayerCount;
        public float LearningRate => Network.LearningRate;
        public Layer OutputLayer => Network.OutputLayer;
        public ILossFunction LossFunction => Network.LossFunction;
        public List<Layer> Layers => Network.Layers;

        public MatrixBase[] WeightGradient { get; }
        public ColumnVectorBase[] BiasGradient { get; }


        public RenderContext(GeneralFeedForwardANN network, int batchSize, ITrainingSet trainingSet)
        {
            this.CurrentThreadID = Thread.CurrentThread.ManagedThreadId;
            this.Network = network;
            this.BatchSize = batchSize;
            this.TrainingSet = trainingSet;
            this.Sigma = new ColumnVectorBase[this.LayerCount];
            this.WeightGradient = new MatrixBase[this.LayerCount];
            this.BiasGradient = new ColumnVectorBase[this.LayerCount];
            this.ActivationContext = new ColumnVectorBase[this.LayerCount];
            this.DerivativeContext = new ColumnVectorBase[this.LayerCount];
        }

        private void SetLastActivation(int layerIndex, ColumnVectorBase lastActivation)
        {
            Debug.Assert(layerIndex >= 0);
            Debug.Assert(lastActivation != null);
            Debug.Assert(ActivationContext[layerIndex] == null);
            ActivationContext[layerIndex] = lastActivation;
        }

        private void SetlayerSigma(int layerIndex, ColumnVectorBase sigma)
        {
            Debug.Assert(this.Sigma[layerIndex] == null);
            this.Sigma[layerIndex] = sigma;
        }

        private void SetLastDerivative(int myLayerIndex, ColumnVectorBase derivative)
        {
            Debug.Assert(DerivativeContext[myLayerIndex] == null);
            DerivativeContext[myLayerIndex] = derivative;   
        }

        public void EpochTrain(int numEpochs)
        {
            for (int i = 0; i < numEpochs; i++)
                BatchTrain(i);
        }

        /// <summary>
        /// Performs mini-batch gradient descent training.
        /// For each batch:
        /// 1. Resets gradient accumulators on all layers
        /// 2. Processes batchSize samples, accumulating gradients
        /// 3. Averages gradients and updates weights once per batch
        /// 
        /// Note: By default uses single-threaded execution for thread safety.
        /// </summary>
        public void BatchTrain(int epochNum, bool doParallel = false)
        {
            bool do2dImage = false;
            List<TrainingPair> trainingPairs = TrainingSet.BuildNewRandomizedTrainingList(do2dImage);
            int totalSamples = TrainingSet.NumberOfSamples;
            int maxBatches = totalSamples / BatchSize;

            int currentSampleIndex = 0;

            for (int batchIdx = 0; batchIdx < maxBatches; batchIdx++)
            {
                // Reset accumulators at start of each batch
                foreach (Layer layer in Network.Layers)
                {
                    layer.ResetAccumulators();
                }

                // Process batchSize samples
                // Each sample: forward pass + backward pass to accumulate gradients
                int batchStartIndex = currentSampleIndex;


                if (doParallel)
                {
                    // Parallel execution (faster but may have issues with CNN layers that store state)
                    Parallel.For(0, BatchSize, sampleIdx =>
                    {
                        // Get the training pair for this sample (thread-safe)
                        TrainingPair trainingPair;
                        int sampleIndex = batchStartIndex + sampleIdx;
                        lock (trainingPairs)
                        {
                            trainingPair = trainingPairs[sampleIndex];
                        }

                        // Forward pass
                        ColumnVectorBase predictedOut = FeedForward(trainingPair.Input);

                        // Backward pass - accumulates gradients into shared layer accumulators
                        BackProp(trainingPair, predictedOut);
                    });
                }
                else
                {
                    // Single-threaded execution (safer, works with all layer types including CNN)
                    for (int sampleIdx = 0; sampleIdx < BatchSize; sampleIdx++)
                    {
                        // Get the training pair for this sample
                        TrainingPair trainingPair;
                        int sampleIndex = batchStartIndex + sampleIdx;
                        trainingPair = trainingPairs[sampleIndex];

                        // Forward pass
                        ColumnVectorBase predictedOut = FeedForward(trainingPair.Input);

                        // Backward pass - accumulates gradients into shared layer accumulators
                        BackProp(trainingPair, predictedOut);
                    }
                }


                // Update weights once per batch using averaged gradients
                foreach (Layer layer in Network.Layers)
                {
                    layer.UpdateWeightsAndBiasesWithScaledGradients(LearningRate);
                }

                // Log progress every 100 batches
                if (batchIdx % 100 == 0)
                {
                    // Use the last sample of this batch for loss calculation
                    TrainingPair sampleForLoss = trainingPairs[currentSampleIndex + BatchSize - 1];
                    ColumnVectorBase predictedOut = FeedForward(sampleForLoss.Input);
                    float totalLoss = Network.GetTotallLoss(sampleForLoss, predictedOut);
                    Console.WriteLine($"Epoch {epochNum}, batch size:{BatchSize}. Finished Batch {batchIdx} with total loss = {totalLoss}");
                }

                currentSampleIndex += BatchSize;
            }
        }

        public ColumnVectorBase FeedForward(Tensor input)
        {
            Tensor lastOutput = input;
            foreach (Layer layer in Layers)            
                lastOutput = layer.FeedFoward(lastOutput);
            
            return lastOutput.ToColumnVector();
        }

        /// <summary>
        /// Accumulates gradients directly into the shared network layers
        /// </summary>
        public void BackProp(TrainingPair trainingPair, ColumnVectorBase predictedOut)
        {
            Tensor dE_dX = Network.LossFunction.Derivative(trainingPair.Output.ToColumnVector(), predictedOut).ToTensor();
            foreach (Layer layer in Network.Layers.Reverse<Layer>())
                dE_dX = layer.BackPropagation(dE_dX);
        }

        public void BackProp_verboseDebug(TrainingPair trainingPair, ColumnVectorBase predictedOut)
        {
            bool debugMode = Environment.GetEnvironmentVariable("NEURALNET_DEBUG") == "1";

            Tensor dE_dX = LossFunction.Derivative(trainingPair.Output.ToColumnVector(), predictedOut).ToTensor();
            if (debugMode)
            {
                Console.WriteLine($"\n[RenderContext.BackProp] Initial dE/dX (loss derivative): [{string.Join(", ", Enumerable.Range(0, dE_dX.ToColumnVector().Size).Select(i => dE_dX.ToColumnVector()[i].ToString("F6")))}]");
            }

            int layerIndex = this.Layers.Count - 1;
            foreach (Layer layer in this.Layers.Reverse<Layer>())
            {
                if (debugMode)
                {
                    Console.WriteLine($"\n[RenderContext.BackProp] Processing layer {layerIndex} ({layer.GetType().Name})");
                }

                if (layer is IActivationFunction)
                {
                    if (debugMode)
                    {
                        var lastAct = (layer as IActivationFunction).LastActivation;
                        Console.WriteLine($"  Passing LastActivation to ReLU: [{string.Join(", ", Enumerable.Range(0, lastAct.ToColumnVector().Size).Select(i => lastAct.ToColumnVector()[i].ToString("F6")))}]");
                        Console.WriteLine($"  Current dE/dX before ReLU: [{string.Join(", ", Enumerable.Range(0, dE_dX.ToColumnVector().Size).Select(i => dE_dX.ToColumnVector()[i].ToString("F6")))}]");
                    }

                    dE_dX = layer.BackPropagation(dE_dX);

                    if (debugMode)
                    {
                        Console.WriteLine($"  After multiplying by dE/dX: [{string.Join(", ", Enumerable.Range(0, dE_dX.ToColumnVector().Size).Select(i => dE_dX.ToColumnVector()[i].ToString("F6")))}]");
                    }
                }
                else
                {
                    if (debugMode)
                    {
                        Console.WriteLine($"  Passing dE/dX to WeightedLayer: [{string.Join(", ", Enumerable.Range(0, dE_dX.ToColumnVector().Size).Select(i => dE_dX.ToColumnVector()[i].ToString("F6")))}]");
                    }

                    dE_dX = layer.BackPropagation(dE_dX);

                    if (debugMode)
                    {
                        Console.WriteLine($"  WeightedLayer returned dE/dX for previous layer: [{string.Join(", ", Enumerable.Range(0, dE_dX.ToColumnVector().Size).Select(i => dE_dX.ToColumnVector()[i].ToString("F6")))}]");
                    }
                }

                layerIndex--;
            }
        }

        public void ScaleAndUpdateWeightsBiasesHelper(int L)
        {
            this.Layers[L].UpdateWeightsAndBiasesWithScaledGradients(LearningRate);
        }

    }
}
