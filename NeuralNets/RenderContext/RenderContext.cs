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

        //
        // do the work
        //
#if true

        public void EpochTrain(int numEpochs)
        {
            // scope of training enumerator is entire epoch
            for (int i = 0; i < numEpochs; i++)
            {
                RenderContext.BatchTrain(this, i);
            }
        }
        /// <summary>
        /// Performs mini-batch gradient descent training.
        /// For each batch:
        /// 1. Resets gradient accumulators on all layers
        /// 2. Processes batchSize samples, accumulating gradients
        /// 3. Averages gradients and updates weights once per batch
        /// 
        /// Note: By default uses single-threaded execution for thread safety with CNN layers.
        /// To enable parallel execution, define PARALLEL_BATCH_TRAIN compilation symbol.
        /// </summary>
        public static void BatchTrain(RenderContext parentContext, int epochNum)
        {
            bool do2dImage = true;
            List<TrainingPair> trainingPairs = parentContext.TrainingSet.BuildNewRandomizedTrainingList(do2dImage);
            int totalSamples = parentContext.TrainingSet.NumberOfSamples;
            int maxBatches = totalSamples / parentContext.BatchSize;
            
            int currentSampleIndex = 0;
            
            for (int batchIdx = 0; batchIdx < maxBatches; batchIdx++)
            {
                // Reset accumulators at start of each batch
                foreach (Layer layer in parentContext.Network.Layers)
                {
                    layer.ResetAccumulators();
                }

                // Process batchSize samples
                // Each sample: forward pass + backward pass to accumulate gradients
                int batchStartIndex = currentSampleIndex;
                
#if PARALLEL_BATCH_TRAIN
                // Parallel execution (faster but may have issues with CNN layers that store state)
                Parallel.For(0, parentContext.BatchSize, sampleIdx =>
                {
                    // Get the training pair for this sample (thread-safe)
                    TrainingPair trainingPair;
                    int sampleIndex = batchStartIndex + sampleIdx;
                    lock (trainingPairs)
                    {
                        trainingPair = trainingPairs[sampleIndex];
                    }
                    
                    // Forward pass
                    ColumnVectorBase predictedOut = FeedForwardStatic(parentContext.Network.Layers, trainingPair.Input);
                    
                    // Backward pass - accumulates gradients into shared layer accumulators
                    BackPropStatic(parentContext.Network, trainingPair, predictedOut);
                });
#else
                // Single-threaded execution (safer, works with all layer types including CNN)
                for (int sampleIdx = 0; sampleIdx < parentContext.BatchSize; sampleIdx++)
                {
                    // Get the training pair for this sample
                    TrainingPair trainingPair;
                    int sampleIndex = batchStartIndex + sampleIdx;
                    trainingPair = trainingPairs[sampleIndex];
                    
                    // Forward pass
                    ColumnVectorBase predictedOut = FeedForwardStatic(parentContext.Network.Layers, trainingPair.Input);
                    
                    // Backward pass - accumulates gradients into shared layer accumulators
                    BackPropStatic(parentContext.Network, trainingPair, predictedOut);
                }
#endif

                // Update weights once per batch using averaged gradients
                foreach (Layer layer in parentContext.Network.Layers)
                {
                    layer.UpdateWeightsAndBiasesWithScaledGradients(parentContext.LearningRate);
                }

                // Log progress every 100 batches
                if (batchIdx % 100 == 0)
                {
                    // Use the last sample of this batch for loss calculation
                    TrainingPair sampleForLoss = trainingPairs[currentSampleIndex + parentContext.BatchSize - 1];
                    ColumnVectorBase predictedOut = FeedForwardStatic(parentContext.Network.Layers, sampleForLoss.Input);
                    float totalLoss = parentContext.Network.GetTotallLoss(sampleForLoss, predictedOut);
                    Console.WriteLine($"Epoch {epochNum}, batch size:{parentContext.BatchSize}. Finished Batch {batchIdx} with total loss = {totalLoss}");
                }

                currentSampleIndex += parentContext.BatchSize;
            }
        }

        /// <summary>
        /// Static version of FeedForward that doesn't require creating a RenderContext
        /// </summary>
        private static ColumnVectorBase FeedForwardStatic(List<Layer> layers, Tensor input)
        {
            Tensor lastOutput = input;
            foreach (Layer layer in layers)
            {
                lastOutput = layer.FeedFoward(lastOutput);
            }
            return lastOutput.ToColumnVector();
        }

        /// <summary>
        /// Static version of BackProp that doesn't require creating a RenderContext
        /// Accumulates gradients directly into the shared network layers
        /// </summary>
        private static void BackPropStatic(NeuralNetworkAbstract network, TrainingPair trainingPair, ColumnVectorBase predictedOut)
        {
            Tensor dE_dX = network.LossFunction.Derivative(trainingPair.Output.ToColumnVector(), predictedOut).ToTensor();
            foreach (Layer layer in network.Layers.Reverse<Layer>())
            {
                // All layers (including activation) handle their own derivative computation
                dE_dX = layer.BackPropagation(dE_dX);
            }
        }
        public void ScaleAndUpdateWeightsBiasesHelper(int L)
        {
            this.Layers[L].UpdateWeightsAndBiasesWithScaledGradients(LearningRate);
        }
#endif


        // temp changes to hack this, undo
        public ColumnVectorBase FeedForward(Tensor inputVecTensor)
        {
            //ColumnVectorBase inputVec = inputVecTensor.ToAvxColumnVector();
            //Debug.Assert(inputVec.Size == this.InputDim);

            Tensor lastOutput = inputVecTensor;
            for (int i = 0; i < this.LayerCount; i++)
            {
                lastOutput = Layers[i].FeedFoward(lastOutput);
                //this.SetLastActivation(i, lastOutput);
            }
            return lastOutput.ToColumnVector();
        }

        // for reference
#if false
        public ColumnVectorBase FeedForward_(Tensor inputVecTensor)
        {
            ColumnVectorBase inputVec = (inputVecTensor as AnnTensor).ColumnVector;
            Debug.Assert(inputVec.Size == this.InputDim);
            ColumnVectorBase prevActivation = inputVec;
            for (int i = 0; i < this.LayerCount; i++)
            {
                WeightedLayer currentLayer = Layers[i] as WeightedLayer;
                MatrixBase w1 = new MatrixBase(currentLayer.Weights.Mat);
                ColumnVectorBase pa = new ColumnVectorBase(prevActivation.Column);
                ColumnVectorBase z1 = new MatrixBase(currentLayer.Weights.Mat) * new ColumnVectorBase(prevActivation.Column);
                ColumnVectorBase z12 = z1 + new ColumnVectorBase(currentLayer.Biases.Column);
                prevActivation = currentLayer.Activate(z12);
                this.SetLastActivation(i, prevActivation);
            }
            return prevActivation;
        }
#endif

// Note: this is specialized for 2 layers (input, hidden, output). Great as a reference
// But not generalized for many layers.
// Great for validation because we know it works.
/*
 * public void BackProp_2layer(TrainingPair trainingPair, ColumnVectorBase predictedOut)
{
    // Second: find W0 - Wn in the hidden layer, just before the output layer
    // We want the derivative of the Error function in terms of the weights (w0 ... wn)
    // d(E)/dw1 = d(E)/o2 * d(o2)/z2 * d(z2)/w = (a-b) * layer.derivative * o1
    // <matrix form> ==> 
    //          (pred - actual)_vec * sigmoid_derivate_vec * layer-1.output_vec
    // d(E)/db = d(E)/o2 * d(o2)/z2 * d(z2)/b
    // 

    WeightedLayer hiddenLayer = WeightedLayers[0];
    WeightedLayer outputLayer = WeightedLayers[1];

    // partial product, before we start the per-w differentials.
    ColumnVectorBase LossPartial = this.LossFunction.Derivative(trainingPair.Output, predictedOut);
    ColumnVectorBase ActivationPartial = outputLayer.Derivative();  // sigmoid partial derivative
    ColumnVectorBase w2_sigma = LossPartial * ActivationPartial;

    // Remember that the weights in the weight matrix are ROWS ...
    // so the dot product of row1 and output vector or activation vector minus the bias is = Z (the input to the activation function)

    // so the gradient w' matrix needs to be rows of gradient weights (or weight deltas) that we get from all the partial derivative shenanigans
    Matrix scaledGradientWeights_outputLayer = this.TrainingRate * BuildGradientWeights(ctx.ActivationContext[1], w2_sigma);
    ColumnVectorBase b2_delta = this.TrainingRate * w2_sigma * 1.0;


    // ----
    // For hidden layer:
    // v = the weights before the hidden layer.
    // bb = biases before the hidden layer.
    // Zl = the input to this node
    // Ol = output = Relu(Zl)
    // in = input n.
    // Zl = v1 * i1 + v2 * i2 + ... 
    // Full D(E)/dv = D(zl)/d(v) * d(Ol)/d(zl) * SUM_OVER_ALL_OUTGOING_EDGES[ D(E)/D(Ol) ]   (for example de1/dzl + de0/dzl + de2/dzl ... deN/dzl)

    // a = output of sigmoid on out put layer
    // z = input to sigmoing on outputlayer
    // E = error at that node on output layer
    // w = weight on edge between hiddend and output layer
    //  D(E)/D(Ol) == D(E)/D(a) * D(a)/Dz * Dz / D(Ol) = (predicted - actual) * sigmoid_derivative(z) * w
    // and the left side:  D(E)/dv = D(zl)/d(v) * d(Ol)/d(zl)
    //                             = i          * Relu'(zl)

    // 
    // Sigma = D(E)/Da * Da / Dz  [ on the output layer). a is the output of sigmoid. z is the input
    // Now multiply sigma by the existing weight matrix:
    // ** from above **  D(E)/D(Ol) == D(E)/D(a) * D(a)/Dz * Dz / D(Ol) = (predicted - actual) * sigmoid_derivative(z) * w
    ColumnVectorBase sum_over_all_de_dOl = outputLayer.Weights.GetTransposedMatrix() * w2_sigma;
    // NOTE: each entry of this column vector as the SUM_OVER_ALL_OUTGOING_EDGES for each HiddenLayer node.
    // for node 3, de_dOl[2] == the sum of all outgoing edges partial derivatives

    // partial weights 
    // ColumnVectorBase DZl_Dv_times_dOl_dZl = trainingPair.Input * hiddenLayer.GetActivationFunctionDerivative();
    ColumnVectorBase DOl_DZL = hiddenLayer.Derivative(ctx, 0) * sum_over_all_de_dOl;
    Matrix scaledGradientWeights_hiddenLayer = this.TrainingRate * BuildGradientWeights(trainingPair.Input, DOl_DZL);

    // partial biases full equation:
    // D(E)/D(bb) = D(zl)/D(bb) * d(Ol)/d(zl) * SUM_OVER_ALL_OUTGOING_EDGES[ D(E)/D(Ol) ]   (for example de1/dzl + de0/dzl + de2/dzl ... deN/dzl)
    // Note all the terms are the same except the first : dzl/dbb
    ColumnVectorBase b1_delta = this.TrainingRate * DOl_DZL * 1.0;

    // UPDATE THESE WEIGHTS AFTER BACK PROP IS DONE
    // Now: Update W2 weight matrix with w2_delta (and same for b)
    outputLayer.AccumulateGradients(scaledGradientWeights_outputLayer, b2_delta);
    outputLayer.UpdateWeightsAndBiases();

    hiddenLayer.AccumulateGradients(scaledGradientWeights_hiddenLayer, b1_delta);
    hiddenLayer.UpdateWeightsAndBiases();
}
*/

        public void BackProp(TrainingPair trainingPair, ColumnVectorBase predictedOut)
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


        private void SetLayerGradients(int L, MatrixBase weightGradient, ColumnVectorBase biasGradient)
        {
            this.BiasGradient[L] = biasGradient;
            this.WeightGradient[L] = weightGradient;
        }

        private MatrixBase BuildGradientWeightsHelper_naive(ColumnVectorBase lastActivation, ColumnVectorBase sigma)
        {
            // Do outer product
            // we want all the sigmas (on the right) times all the Outpus (from the left) to look like the wiehgt matrix
            // where the top row represents the weights of the entier (left) layer.
            // this matrix is the D(E)/D(w) final result, and the partial derivative of the massive dot product at each right node, per neft node is simply the output of the left node
            //sigma = this.TrainingRate * sigma;
            // Matrix scaledGradientWeights = sigma * lastActivation.Transpose();

            // outer product:  o1s1 o2s1 o3s1 o4s1 ... onS1    o1s2 o2s2 o3s2 ... oNs2 
//            MatrixBase gradientDelta = sigma * lastActivation.Transpose();
            MatrixBase gradientDelta = sigma.OuterProduct(lastActivation);
            return gradientDelta;
        }

        private MatrixBase BuildGradientWeightsHelper(ColumnVectorBase lastActivation, ColumnVectorBase sigma)
        {
            MatrixBase gradientDelta = sigma.OuterProduct(lastActivation);
            return gradientDelta;
        }

    }
}
