using MatrixLibrary;
using System.Diagnostics;

namespace NeuralNets
{
    public class RenderContext
    {
        public int BatchSize { get; }
        public GeneralFeedForwardANN Network { get; }
        public int CurrentThreadID { get; private set; }
        public ColumnVector[] Sigma { get; private set; }
        public ColumnVector[] ActivationContext { get; }
        public ColumnVector[] DerivativeContext { get; }
        public bool DoRandomSamples { get; private set; }
        public virtual ITrainingSet TrainingSet { get; }

        public int InputDim => Network.InputDim;
        public int OutputDim => Network.OutputDim;
        public int LayerCount => Network.LayerCount;
        public float TrainingRate => Network.TrainingRate;
        public WeightedLayer OutputLayer => Network.OutputLayer;
        public ILossFunction LossFunction => Network.LossFunction;
        public List<WeightedLayer> WeightedLayers => Network.WeightedLayers;

        public Matrix2D[] WeightGradient { get; }
        public ColumnVector[] BiasGradient { get; }


        public RenderContext(GeneralFeedForwardANN network, int batchSize, ITrainingSet trainingSet)
        {
            this.CurrentThreadID = Thread.CurrentThread.ManagedThreadId;
            this.Network = network;
            this.BatchSize = batchSize;
            this.TrainingSet = trainingSet;
            this.Sigma = new ColumnVector[this.LayerCount];
            this.WeightGradient = new Matrix2D[this.LayerCount];
            this.BiasGradient = new ColumnVector[this.LayerCount];
            this.ActivationContext = new ColumnVector[this.LayerCount];
            this.DerivativeContext = new ColumnVector[this.LayerCount];
        }

        private void SetLastActivation(int layerIndex, ColumnVector lastActivation)
        {
            Debug.Assert(layerIndex >= 0);
            Debug.Assert(lastActivation != null);
            Debug.Assert(ActivationContext[layerIndex] == null);
            ActivationContext[layerIndex] = lastActivation;
        }

        private void SetlayerSigma(int layerIndex, ColumnVector sigma)
        {
            Debug.Assert(this.Sigma[layerIndex] == null);
            this.Sigma[layerIndex] = sigma;
        }

        private void SetLastDerivative(int myLayerIndex, ColumnVector derivative)
        {
            Debug.Assert(DerivativeContext[myLayerIndex] == null);
            DerivativeContext[myLayerIndex] = derivative;   
        }

        //
        // do the work
        //

        public void EpochTrain(int numEpochs)
        {
            // scope of training enumerator is entire epoch
            for (int i = 0; i < numEpochs; i++)
            {
                RenderContext.BatchTrain(this, i);
            }
        }

        public static void BatchTrain(RenderContext parentContext, int epochNum)
        {
            // multiple feed forward, random samples from training set
            // then run a backprop based on average O and average L
            // average all gradients then update biases/weights


            WeightedLayer outputLayer = parentContext.OutputLayer;
            List<TrainingPair> trainingPairs = parentContext.TrainingSet.BuildNewRandomizedTrainingList();
            int currentTP = 0;
            int batchCount = 0;
            int totalSamples = parentContext.TrainingSet.NumberOfSamples;
            int maxBatches = totalSamples / parentContext.BatchSize;
            for (int j = 0; j < maxBatches; j++)
            {
                TrainingPair trainingPair = null;
                ColumnVector predictedOut = null;

                int loopsPerThread = 32;
                if(parentContext.BatchSize / loopsPerThread < 16)
                {
                    loopsPerThread = 1;
                }

                // debug only
                int actualBatchSize = 0;

                // single entry on this guy
                List<RenderContext> renderContexts = new List<RenderContext>();

                // debug only
                int a = Thread.CurrentThread.ManagedThreadId;

                Object thisLock = new object();
                // 
                //for (int i = 0; i < parentContext.BatchSize; i++)
                Parallel.For(0, parentContext.BatchSize / loopsPerThread, i =>
                {
                    List<RenderContext> perCoreRenderContexts = new List<RenderContext>();
                    //Console.WriteLine($"Launched batch instance {i} with thread id {Thread.CurrentThread.ManagedThreadId}");
                    for (int j = 0; j < loopsPerThread; j++)
                    {
                        RenderContext ctx = new RenderContext(parentContext.Network, 0, null); // todo: hmm, mabe need to specialize the render context??
                        trainingPair = trainingPairs[currentTP++];
                        predictedOut = ctx.FeedForward(trainingPair.Input);
                        ctx.BackProp(trainingPair, predictedOut);
                        perCoreRenderContexts.Add(ctx);
                    }
                    // LOCK
                    lock (thisLock)
                    {
                        renderContexts.AddRange(perCoreRenderContexts);
                        actualBatchSize++;
                    }
                    // UNLOCK
                });
                Debug.Assert(actualBatchSize == parentContext.BatchSize);

                // All the RenderContexts above (in the render batch) now have their own gradients
                // Sum and average all gradients

                Matrix2D[] weightGradients = new Matrix2D[parentContext.LayerCount];
                ColumnVector[] biasGradients = new ColumnVector[parentContext.LayerCount];
                // accumulate all weight and bias gradients
                // ie: For every context, for every layer: accumulate the gradient by layer into one sum, ditto biases.
                for (int rc = 0; rc < renderContexts.Count; rc++)
                {
                    for (int L = 0; L < parentContext.LayerCount; L++)
                    {
                        weightGradients[L] = weightGradients[L] == null ? 
                            renderContexts[rc].WeightGradient[L] : 
                            weightGradients[L] + renderContexts[rc].WeightGradient[L];

                        biasGradients[L] = biasGradients[L] == null ?
                            renderContexts[rc].BiasGradient[L] :
                            biasGradients[L] + renderContexts[rc].BiasGradient[L];
                    }
                }

                Debug.Assert(a == Thread.CurrentThread.ManagedThreadId);

                // scale by 1/batchsize (ie: average) & learning rate at the same time
                for (int L = 0; L < parentContext.LayerCount; L++)
                {
                    float scaleFactor = parentContext.TrainingRate / (float)parentContext.BatchSize;
                    biasGradients[L] *= scaleFactor;
                    weightGradients[L] *= scaleFactor;

                    // Now update each layer in the network by the accumulated, averaged, and scaled weight and bias gradients
                    WeightedLayer layer = parentContext.WeightedLayers[L];
                    layer.UpdateWeightsAndBiasesWithScaledGradients(weightGradients[L], biasGradients[L]);
                }

                // finished batch N
                if (batchCount % 100 == 0)
                {
                    RenderContext ctx = new RenderContext(parentContext.Network, 0, null); 
                    predictedOut = ctx.FeedForward(trainingPair.Input);
                    float totalLoss = ctx.Network.GetTotallLoss(trainingPair, predictedOut);
                    Console.WriteLine($"Epoch {epochNum}, batch size:{parentContext.BatchSize}. Finished Batch {batchCount} with total loss = {totalLoss}");
                }
                batchCount++;
            }

            // remember that we do a real backprop per trianing case, but we just don't adjust the weights and biases
            // instead we remember all the W and B gradient matrices and vectors. add them all up. get average. and then adjust all weights
        }

        public void ScaleAndUpdateWeightsBiasesHelper(int L)
        {
            float scaleFactor = this.TrainingRate / (float)this.BatchSize;
            this.BiasGradient[L] *= scaleFactor;
            this.WeightGradient[L] *= scaleFactor;
            this.WeightedLayers[L].UpdateWeightsAndBiasesWithScaledGradients(this.WeightGradient[L], this.BiasGradient[L]);
        }

        public ColumnVector FeedForward_naive(ColumnVector inputVec)
        {
            Debug.Assert(inputVec.Size == this.InputDim);
            ColumnVector prevActivation = inputVec;
            for (int i = 0; i < this.LayerCount; i++)
            {
                WeightedLayer currentLayer = WeightedLayers[i];
                ColumnVector z1 = currentLayer.Weights * prevActivation;
                ColumnVector z12 = z1 + currentLayer.Biases;
                prevActivation = currentLayer.Activate(z12);
                this.SetLastActivation(i, prevActivation);
            }
            return prevActivation;
        }

        public ColumnVector FeedForward(ColumnVector inputVec)
        {
            Debug.Assert(inputVec.Size == this.InputDim);
            ColumnVector prevActivation = inputVec;
            for (int i = 0; i < this.LayerCount; i++)
            {
                WeightedLayer currentLayer = WeightedLayers[i];
                AvxMatrix w1 = new AvxMatrix(currentLayer.Weights.Mat);
                AvxColumnVector pa = new AvxColumnVector(prevActivation.Column);
                AvxColumnVector z1 = new AvxMatrix(currentLayer.Weights.Mat) * new AvxColumnVector(prevActivation.Column);
                AvxColumnVector z12 = z1 + new AvxColumnVector(currentLayer.Biases.Column);
                prevActivation = currentLayer.Activate(new ColumnVector(z12.Column));
                this.SetLastActivation(i, prevActivation);
            }
            return prevActivation;
        }

        // Note: this is specialized for 2 layers (input, hidden, output). Great as a reference
        // But not generalized for many layers.
        // Great for validation because we know it works.
        /*
         * public void BackProp_2layer(TrainingPair trainingPair, ColumnVector predictedOut)
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
            ColumnVector LossPartial = this.LossFunction.Derivative(trainingPair.Output, predictedOut);
            ColumnVector ActivationPartial = outputLayer.Derivative();  // sigmoid partial derivative
            ColumnVector w2_sigma = LossPartial * ActivationPartial;

            // Remember that the weights in the weight matrix are ROWS ...
            // so the dot product of row1 and output vector or activation vector minus the bias is = Z (the input to the activation function)

            // so the gradient w' matrix needs to be rows of gradient weights (or weight deltas) that we get from all the partial derivative shenanigans
            Matrix scaledGradientWeights_outputLayer = this.TrainingRate * BuildGradientWeights(ctx.ActivationContext[1], w2_sigma);
            ColumnVector b2_delta = this.TrainingRate * w2_sigma * 1.0;


            // ----
            // For hidden layer:
            // v = the weights before the hiddne layer.
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
            ColumnVector sum_over_all_de_dOl = outputLayer.Weights.GetTransposedMatrix() * w2_sigma;
            // NOTE: each entry of this column vector as the SUM_OVER_ALL_OUTGOING_EDGES for each HiddenLayer node.
            // for node 3, de_dOl[2] == the sum of all outgoing edges partial derivatives

            // partial weights 
            // ColumnVector DZl_Dv_times_dOl_dZl = trainingPair.Input * hiddenLayer.GetActivationFunctionDerivative();
            ColumnVector DOl_DZL = hiddenLayer.Derivative(ctx, 0) * sum_over_all_de_dOl;
            Matrix scaledGradientWeights_hiddenLayer = this.TrainingRate * BuildGradientWeights(trainingPair.Input, DOl_DZL);

            // partial biases full equation:
            // D(E)/D(bb) = D(zl)/D(bb) * d(Ol)/d(zl) * SUM_OVER_ALL_OUTGOING_EDGES[ D(E)/D(Ol) ]   (for example de1/dzl + de0/dzl + de2/dzl ... deN/dzl)
            // Note all the terms are the same except the first : dzl/dbb
            ColumnVector b1_delta = this.TrainingRate * DOl_DZL * 1.0;

            // UPDATE THESE WEIGHTS AFTER BACK PROP IS DONE
            // Now: Update W2 weight matrix with w2_delta (and same for b)
            outputLayer.AccumulateGradients(scaledGradientWeights_outputLayer, b2_delta);
            outputLayer.UpdateWeightsAndBiases();

            hiddenLayer.AccumulateGradients(scaledGradientWeights_hiddenLayer, b1_delta);
            hiddenLayer.UpdateWeightsAndBiases();
        }
        */

        //
        // generalized back propagation for many layers
        //
        public void BackProp_naive( TrainingPair trainingPair, ColumnVector predictedOut)
        {
            //
            // Special case the Output Layer
            //
            int outputLayerIndex = LayerCount - 1;
            int secondToLastLayerIndex = LayerCount - 2;
            WeightedLayer outputLayer = WeightedLayers[outputLayerIndex];
            WeightedLayer secondToLastLayer = WeightedLayers[secondToLastLayerIndex];

            ColumnVector outputLayerSigma;
            // special case softmax and cross entropy
            if (outputLayer.IsSoftMaxActivation)
            {
                Debug.Assert(this.LossFunction is CategoricalCrossEntropy ||
                             this.LossFunction is SparseCategoricalCrossEntropy ||
                             this.LossFunction is VanillaCrossEntropy);

                // after all the crazy derivatives of softmax * crossentrotpy, we just end up with: a - y
                // which is 'activtion' of softmax minus the truth vector.  must be onehot encoded
                outputLayerSigma = ActivationContext[outputLayerIndex] - trainingPair.Output;
            }
            else
            {
                // partial product, before we start the per-w differentials.
                ColumnVector LossPartial = this.LossFunction.Derivative(trainingPair.Output, predictedOut);
                ColumnVector ActivationPartial = outputLayer.Derivative(this.ActivationContext[outputLayerIndex]);    // outputLayer.ActivationFunction.Derivative();
                this.SetLastDerivative(outputLayerIndex, ActivationPartial);
                outputLayerSigma = LossPartial * ActivationPartial;
            }

            Matrix2D outputWeightGradient = BuildGradientWeightsHelper_naive(this.ActivationContext[LayerCount - 2], outputLayerSigma);
            ColumnVector outputBiasGradient = outputLayerSigma * 1;

            this.SetlayerSigma(outputLayerIndex, outputLayerSigma);
            this.SetLayerGradients(outputLayerIndex, outputWeightGradient, outputBiasGradient);

            //
            // Now do back prop through all the hidden layers (ie: not the output). Special case for the input layer. That's why we go all the way to zero.
            //
            for (int L = LayerCount - 2; L >= 0; L--)
            {
                WeightedLayer currentLayer = WeightedLayers[L];
                WeightedLayer layerToTheRight = WeightedLayers[L + 1];

                ColumnVector layerToTheRightSigma = this.Sigma[L + 1];
                ColumnVector sum_over_all_de_dOl = layerToTheRight.Weights.GetTransposedMatrix() * layerToTheRightSigma;
                ColumnVector DOl_DZL = currentLayer.Derivative(this.ActivationContext[L]) * sum_over_all_de_dOl;
                this.SetlayerSigma(L, DOl_DZL);

                // If we're the first hidden layer, L=0, then the last activation is the input from the input layer
                // which isn't represented as a layer. But could be.
                ColumnVector activationToTheLeft;
                if (L == 0)
                {
                    // no one to the left, use input from training
                    activationToTheLeft = trainingPair.Input;
                }
                else
                {
                    activationToTheLeft = this.ActivationContext[L - 1];
                }

                Matrix2D currentLayerGradientWeights = BuildGradientWeightsHelper_naive(activationToTheLeft, DOl_DZL);
                ColumnVector currentLayerGradientBias = DOl_DZL;
                this.SetLayerGradients(L, currentLayerGradientWeights, currentLayerGradientBias);
            }
        }

        // avx accelerated
        public void BackProp(TrainingPair trainingPair, ColumnVector predictedOut)
        {
            //
            // Special case the Output Layer
            //
            int outputLayerIndex = LayerCount - 1;
            int secondToLastLayerIndex = LayerCount - 2;
            WeightedLayer outputLayer = WeightedLayers[outputLayerIndex];
            WeightedLayer secondToLastLayer = WeightedLayers[secondToLastLayerIndex];

            AvxColumnVector outputLayerSigma;
            // special case softmax and cross entropy
            if (outputLayer.IsSoftMaxActivation)
            {
                Debug.Assert(this.LossFunction is CategoricalCrossEntropy ||
                             this.LossFunction is SparseCategoricalCrossEntropy ||
                             this.LossFunction is VanillaCrossEntropy);

                // after all the crazy derivatives of softmax * crossentrotpy, we just end up with: a - y
                // which is 'activtion' of softmax minus the truth vector.  must be onehot encoded
                outputLayerSigma = new AvxColumnVector(ActivationContext[outputLayerIndex].Column) - new AvxColumnVector(trainingPair.Output.Column);
            }
            else
            {
                // partial product, before we start the per-w differentials.
                ColumnVector LossPartial = this.LossFunction.Derivative(trainingPair.Output, predictedOut);
                ColumnVector ActivationPartial = outputLayer.Derivative(this.ActivationContext[outputLayerIndex]);
                this.SetLastDerivative(outputLayerIndex, ActivationPartial);
                outputLayerSigma = new AvxColumnVector(LossPartial.Column) * new AvxColumnVector(ActivationPartial.Column);
            }

            AvxMatrix outputWeightGradient = BuildGradientWeightsHelper(new AvxColumnVector(this.ActivationContext[LayerCount - 2].Column), outputLayerSigma);
            AvxColumnVector outputBiasGradient = outputLayerSigma;

            this.SetlayerSigma(outputLayerIndex, new ColumnVector(outputLayerSigma.Column));
            this.SetLayerGradients(outputLayerIndex, new Matrix2D(outputWeightGradient.Mat), new ColumnVector(outputBiasGradient.Column));

            //
            // Now do back prop through all the hidden layers (ie: not the output). Special case for the input layer. That's why we go all the way to zero.
            //
            for (int L = LayerCount - 2; L >= 0; L--)
            {
                WeightedLayer currentLayer = WeightedLayers[L];
                WeightedLayer layerToTheRight = WeightedLayers[L + 1];

                ColumnVector layerToTheRightSigma = this.Sigma[L + 1];
                ColumnVector sum_over_all_de_dOl = layerToTheRight.Weights.GetTransposedMatrix() * layerToTheRightSigma;
                ColumnVector DOl_DZL = currentLayer.Derivative(this.ActivationContext[L]) * sum_over_all_de_dOl;
                this.SetlayerSigma(L, DOl_DZL);

                // If we're the first hidden layer, L=0, then the last activation is the input from the input layer
                // which isn't represented as a layer. But could be.
                ColumnVector activationToTheLeft;
                if (L == 0)
                {
                    // no one to the left, use input from training
                    activationToTheLeft = trainingPair.Input;
                }
                else
                {
                    activationToTheLeft = this.ActivationContext[L - 1];
                }

                AvxMatrix currentLayerGradientWeights = BuildGradientWeightsHelper(activationToTheLeft.ToAvxVector(), DOl_DZL.ToAvxVector());
                ColumnVector currentLayerGradientBias = DOl_DZL;
                this.SetLayerGradients(L, currentLayerGradientWeights.ToMatrix2d(), currentLayerGradientBias);
            }
        }

        private void SetLayerGradients(int L, Matrix2D weightGradient, ColumnVector biasGradient)
        {
            this.BiasGradient[L] = biasGradient;
            this.WeightGradient[L] = weightGradient;
        }

        private Matrix2D BuildGradientWeightsHelper_naive(ColumnVector lastActivation, ColumnVector sigma)
        {
            // Do outer product
            // we want all the sigmas (on the right) times all the Outpus (from the left) to look like the wiehgt matrix
            // where the top row represents the weights of the entier (left) layer.
            // this matrix is the D(E)/D(w) final result, and the partial derivative of the massive dot product at each right node, per neft node is simply the output of the left node
            //sigma = this.TrainingRate * sigma;
            // Matrix scaledGradientWeights = sigma * lastActivation.Transpose();

            // outer product:  o1s1 o2s1 o3s1 o4s1 ... onS1    o1s2 o2s2 o3s2 ... oNs2 
//            Matrix2D gradientDelta = sigma * lastActivation.Transpose();
            Matrix2D gradientDelta = sigma.OuterProduct(lastActivation);
            return gradientDelta;
        }

        private AvxMatrix BuildGradientWeightsHelper(AvxColumnVector lastActivation, AvxColumnVector sigma)
        {
            AvxMatrix gradientDelta = sigma.OuterProduct(lastActivation);
            return gradientDelta;
        }

    }
}
