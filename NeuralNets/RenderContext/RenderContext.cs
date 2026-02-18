using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace NeuralNets
{
    public class RenderContext
    {
        private readonly int batchLogRate = 200; // how often to log progress during batch training (in batches)

        public int BatchSize { get; }
        public GeneralFeedForwardANN Network { get; }
        public int CurrentThreadID { get; private set; }
        public virtual ITrainingSet TrainingSet { get; }

        public int InputDim => Network.InputDim;
        public int OutputDim => Network.OutputDim;
        public int LayerCount => Network.LayerCount;
        public float LearningRate => Network.LearningRate;
        public Layer OutputLayer => Network.OutputLayer;
        public ILossFunction LossFunction => Network.LossFunction;
        public List<Layer> Layers => Network.Layers;

        public RenderContext(GeneralFeedForwardANN network, int batchSize, ITrainingSet trainingSet)
        {
            this.CurrentThreadID = Thread.CurrentThread.ManagedThreadId;
            this.Network = network;
            this.BatchSize = batchSize;
            this.TrainingSet = trainingSet;
        }

        public void EpochTrain(int numEpochs, bool doParallel = false)
        {
            List<RenderContext> contexts = new List<RenderContext>();
            if (doParallel)
            {
                // get number of cores
                int numCores = Environment.ProcessorCount / 2;
                Console.WriteLine($"Processor count: {numCores}. Setting max degree of parallelism to {numCores}.");

                // Create a bunch of cloned RenderContextx.
                // Each will have fresh layers and networks like this main context (this).
                for (int i = 0; i < numCores; i++)
                {
                    GeneralFeedForwardANN networkCopy = DeepCopyNetwork(this.Network);
                    RenderContext contextCopy = new RenderContext(networkCopy, this.BatchSize, this.TrainingSet);
                    contexts.Add(contextCopy);
                }

                for (int i = 0; i < numEpochs; i++)
                    BatchTrain_parallel(i, numCores, contexts, this.TrainingSet);
            }
            else
            {
                for (int i = 0; i < numEpochs; i++)
                    BatchTrain(i, doParallel);
            }

        }

        private GeneralFeedForwardANN DeepCopyNetwork(GeneralFeedForwardANN network)
        {
            // does a deep copy of the network by creating a new instance and copying all layers and their parameters
            GeneralFeedForwardANN networkCopy = new GeneralFeedForwardANN(network);            
            return networkCopy;
        }

        public void BatchTrain_parallel(
            int epochNum,
            int numCores, 
            List<RenderContext> contexts, 
            ITrainingSet trainingSet)
        {
            Debug.Assert(contexts.Count == numCores, "Number of contexts must match number of cores for parallel batch training.");
            RenderContext mainContext = this; // the main context whose network will be updated with averaged weights
            bool do2dImage = false;
            List<TrainingPair> trainingPairs = trainingSet.BuildNewRandomizedTrainingList(do2dImage);
            int totalSamples = trainingSet.NumberOfSamples;
            int maxBatches = totalSamples / BatchSize;
            int batchSize = BatchSize;
            int currentSampleIndex = 0;
            var options = new ParallelOptions
            {
                MaxDegreeOfParallelism = numCores
            };

            // let's say you have 60,000 samples. With a batch size of 100, that's 600 batches per epoch.
            // if you have 8 cores, you could process 8 batches in parallel, which means you'd have 75 sets of parallel batches to process per epoch.
            // as each set of parallel batches completes, you'd average the weights across the 8 contexts before moving on to the next set of parallel batches.

            // execute batches in groups of 'numCores' to maximize parallelism.
            int batchGroups = maxBatches / numCores;
            
            // Timing accumulators for profiling
            long totalCopyTime = 0;
            long totalParallelTime = 0;
            long totalSyncTime = 0;
            int logInterval = 10; // Log timing every N batch groups
            
            for (int batchGroupIdx = 0; batchGroupIdx < batchGroups; batchGroupIdx++)
            {
                // Process batchSize samples
                // Each sample: forward pass + backward pass to accumulate gradients
                int batchStartIndex = currentSampleIndex;

                // Time the weight copy operation
                var copyStopwatch = Stopwatch.StartNew();
                // --
                // Every context has its own network and layers, by design for trhead safety, however, if we don't copy main's weights and biases over
                // then we're learning from stale or divergent w/b on each context.  so start each context with the same weights/biases as main, then they can diverge during the batch, but we are starting from the same place.
                // --
                foreach(RenderContext ctx in contexts)
                {
                    ctx.CopyWeightsAndBiasesFrom(mainContext);
                }
                copyStopwatch.Stop();
                totalCopyTime += copyStopwatch.ElapsedMilliseconds;

                // Time the parallel training section
                var parallelStopwatch = Stopwatch.StartNew();
                int[] threadIds = new int[numCores];
                long[] threadTimes = new long[numCores];
                
                Parallel.For(0, numCores, options, coreIdx =>
                {
                    int threadId = Thread.CurrentThread.ManagedThreadId;
                    threadIds[coreIdx] = threadId;
                    var threadStopwatch = Stopwatch.StartNew();
                    
                    RenderContext context = contexts[coreIdx];
                    foreach (Layer layer in context.Network.Layers)                    
                        layer.ResetAccumulators();                    

                    // Each context processes its own batch of samples
                    int samplesProcessed = 0;
                    for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
                    {
                        int coreSpecificSampleIndex = batchStartIndex + coreIdx * batchSize + sampleIdx;
                        if (coreSpecificSampleIndex >= trainingPairs.Count)
                            break; // safety check to avoid out-of-range

                        TrainingPair trainingPair = trainingPairs[coreSpecificSampleIndex];                        
                        ColumnVectorBase predictedOut = context.FeedForward(trainingPair.Input);
                        context.BackProp(trainingPair, predictedOut);
                        samplesProcessed++;
                    }
                    
                    threadStopwatch.Stop();
                    threadTimes[coreIdx] = threadStopwatch.ElapsedMilliseconds;
                });
                parallelStopwatch.Stop();
                totalParallelTime += parallelStopwatch.ElapsedMilliseconds;
                
                // Log thread diagnostics every N batch groups
                if (batchGroupIdx % logInterval == 0)
                {
                    long minThreadTime = threadTimes.Min();
                    long maxThreadTime = threadTimes.Max();
                    long avgThreadTime = (long)threadTimes.Average();
                    double imbalance = maxThreadTime > 0 ? (double)(maxThreadTime - minThreadTime) / maxThreadTime * 100 : 0;
                    
                    Console.WriteLine($"[THREADS] Core times (ms): {string.Join(", ", threadTimes)}");
                    Console.WriteLine($"[THREADS] Min={minThreadTime}, Max={maxThreadTime}, Avg={avgThreadTime}, Imbalance={imbalance:F1}%");
                    Console.WriteLine($"[THREADS] Thread IDs: {string.Join(", ", threadIds)}");
                }

                // Time the synchronization section
                var syncStopwatch = Stopwatch.StartNew();
                // Update weights once per numCore batches using averaged gradients
                // Every Rendercontext has been accumulating Weights & Biases into its own network layers,
                // so now we need to average those W&B across contexts
                // and update the main network's weights/biases, then update main gradients.

                mainContext.ResetWeightsAndBiasesAccumulatorCounters();
                foreach (RenderContext context in contexts)
                {
                    mainContext.AccumulateWeightsAndBiasesFrom(context);
                }
                mainContext.UpdateGradientsFromAccumulatorsAndReset();
                syncStopwatch.Stop();
                totalSyncTime += syncStopwatch.ElapsedMilliseconds;
                
                // Log timing every N batch groups
                if (batchGroupIdx > 0 && batchGroupIdx % logInterval == 0)
                {
                    long totalTime = totalCopyTime + totalParallelTime + totalSyncTime;
                    double copyPct = totalTime > 0 ? (double)totalCopyTime / totalTime * 100 : 0;
                    double parallelPct = totalTime > 0 ? (double)totalParallelTime / totalTime * 100 : 0;
                    double syncPct = totalTime > 0 ? (double)totalSyncTime / totalTime * 100 : 0;
                    
                    Console.WriteLine($"[TIMING] BatchGroup {batchGroupIdx}: " +
                        $"Copy={totalCopyTime}ms ({copyPct:F1}%), " +
                        $"Parallel={totalParallelTime}ms ({parallelPct:F1}%), " +
                        $"Sync={totalSyncTime}ms ({syncPct:F1}%) " +
                        $"| Parallel/Sync Ratio: {(double)totalParallelTime / totalSyncTime:F2}x");
                    
                    // Reset counters for next interval
                    totalCopyTime = 0;
                    totalParallelTime = 0;
                    totalSyncTime = 0;
                }


                // Log progress every 100 batches
                // Add thread id in here to verify that different threads are processing different batches
                int globalBatchIdx = batchGroupIdx * numCores;
                if (globalBatchIdx % contexts[0].batchLogRate == 0)
                {
                    // Use the last sample of this batch group for loss calculation
                    int lastSampleIdx = Math.Min(currentSampleIndex + (numCores * BatchSize) - 1, trainingPairs.Count - 1);
                    TrainingPair sampleForLoss = trainingPairs[lastSampleIdx];
                    ColumnVectorBase predictedOut = FeedForward(sampleForLoss.Input);
                    float totalLoss = Network.GetTotallLoss(sampleForLoss, predictedOut);
                    Console.WriteLine($"Epoch {epochNum}, batch size:{BatchSize}. Finished Batch Group {batchGroupIdx}/{batchGroups} (batches {globalBatchIdx}-{globalBatchIdx + numCores - 1}) with total loss = {totalLoss}");
                }

                currentSampleIndex += (numCores * BatchSize);
            } // for batchGroupIdx
        }

        private void CopyWeightsAndBiasesFrom(RenderContext mainContext)
        {
            Debug.Assert(this.Layers.Count == mainContext.Layers.Count, "Layer count mismatch when copying weights and biases from main context.");
            for(int i = 0; i < this.Layers.Count; i++)
            {
                Layers[i].CopyWeightsAndBiasesFrom(mainContext.Layers[i]);
            }
        }

        private int numSamples = 0;

        private void AccumulateWeightsAndBiasesFrom(RenderContext context)
        {
            for(int l = 0; l < this.Layers.Count; l++)
            {
                this.Layers[l].AccumulateGradientsFrom(context.Layers[l]);
            }
            numSamples++;
        }

        private void ResetWeightsAndBiasesAccumulatorCounters()
        {
            numSamples = 0;
        }

        private void UpdateGradientsFromAccumulatorsAndReset()
        {
            // Average the accumulated gradients across all contexts (divide by numSamples) and update weights
            foreach (Layer layer in this.Layers)
            {
                if(layer is WeightedLayer || layer is ConvolutionLayer)
                    Debug.Assert(numSamples == layer.AccumulationCount, $"numSamples ({numSamples}) must match the number of accumulated gradients ({layer.AccumulationCount}) for correct averaging.");
                layer.UpdateWeightsAndBiasesWithScaledGradients(LearningRate);
            }

            // reset all layer accumulators, get ready for next epoch.
            foreach (Layer layer in this.Layers)            
                layer.ResetAccumulators();            
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

                // Update weights once per batch using averaged gradients
                foreach (Layer layer in Network.Layers)
                {
                    layer.UpdateWeightsAndBiasesWithScaledGradients(LearningRate);
                }

                // Log progress every 100 batches
                // Add thread id in here to verify that different threads are processing different batches
                if (batchIdx % batchLogRate == 0)
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
