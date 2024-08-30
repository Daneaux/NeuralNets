using NumReaderNetwork;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public class GeneralFeedForwardANN
    {
        public double TrainingRate { get; protected set; }
        public int BatchSize { get; protected set; }
        public int InputDim { get; protected set; }
        public int OutputDim { get; protected set; }
        public List<WeightedLayer> WeightedLayers { get; set; } = [];
        public int LayerCount { get {  return WeightedLayers.Count; } }
        public ILossFunction LossFunction { get; protected set; }
        public WeightedLayer OutputLayer { get { return WeightedLayers[WeightedLayers.Count - 1]; } }

        public bool DoRandomSamples { get; private set; }
        public virtual ITrainingSet TrainingSet { get; protected set; }

        protected GeneralFeedForwardANN(double trainingRate, int batchSize, ITrainingSet trainingSet)
        {
            this.TrainingRate = trainingRate;
            this.BatchSize = batchSize;
            this.TrainingSet = trainingSet;
            this.InputDim = trainingSet.InputDimension;
            this.OutputDim = trainingSet.OutputDimension;
        }

        public GeneralFeedForwardANN(List<WeightedLayer> layers, double trainingRate, int batchSize, ILossFunction lossFunction, ITrainingSet trainingSet) : this(trainingRate, batchSize, trainingSet)
        {
            Debug.Assert(layers != null);
            Debug.Assert(layers.Count > 0);
            lossFunction ??= new SquaredLoss();
            this.LossFunction = lossFunction;
            this.WeightedLayers = layers;
            Debug.Assert(OutputLayer.NumNodes == trainingSet.OutputDimension);
            this.OutputDim = trainingSet.OutputDimension;
        }

        public void EpochTrain(int numEpochs)
        {
            // scope of training enumerator is entire epoch
            for (int i = 0; i < numEpochs; i++)
            {
                this.BatchTrain();
            }
        }

        public void BatchTrain()
        {
            // multiple feed forward, random samples from training set
            // collect all outputs. average them
            // get average loss (how?)
            // then run a backprop based on average O and average L
            // q: what's the right Y (truth) for N samples?
            // we can average the loss. but then the derivative might be wrong for softmax + crossEntropy.

            WeightedLayer outputLayer = this.OutputLayer;
            List<TrainingPair> trainingPairs = this.TrainingSet.BuildNewRandomizedTrainingList();
            int currentTP = 0;
            int batchCount = 0;
            int totalSamples = this.TrainingSet.NumberOfSamples;
            int maxBatches = totalSamples / this.BatchSize;
            for(int j = 0; j < maxBatches; j++)
            {
                TrainingPair trainingPair = null;
                ColumnVector predictedOut = null;

                // debug only
                int actualBatchSize = 0;
                
                for (int i = 0; i < this.BatchSize; i++, actualBatchSize++)
                {
                    trainingPair = trainingPairs[currentTP++];
                    predictedOut = this.FeedForward(trainingPair.Input);
                    this.BackProp(trainingPair, predictedOut);
                }
                Debug.Assert(actualBatchSize == this.BatchSize);

                // using accumualted gradients:
                // scale by 1/batchsize (ie: average) & learning rate
                // scale by learning rate
                for (int i = 0; i < this.WeightedLayers.Count; i++)
                {
                    WeightedLayer layer = this.WeightedLayers[i];
                    double scaleFactor = this.TrainingRate / (double)this.BatchSize;
                    layer.ScaleWeightAndBiasesGradient(scaleFactor);
                }

                // adjust all weights and biases
                this.UpdateWeightsAndBiases();

                // finished batch N
                if (batchCount % 100 == 0)
                {
                    predictedOut = this.FeedForward(trainingPair.Input);
                    double totalLoss = this.GetTotallLoss(trainingPair, predictedOut);
                    Console.WriteLine($"Finished Batch {batchCount} with total loss = {totalLoss}");
                }
                batchCount++;
            }           

            // remember that we do a real backprop per trianing case, but we just don't adjust the weights and biases
            // instead we remember all the W and B gradient matrices and vectors. add them all up. get average. and then adjust all weights
        }

        public ColumnVector FeedForward(ColumnVector inputVec)
        {
            Debug.Assert(inputVec.Size == this.InputDim);
            ColumnVector prevActivation = inputVec;
            for (int i = 0; i < this.LayerCount; i++)
            {
                WeightedLayer currentLayer = WeightedLayers[i];
                ColumnVector z1 = currentLayer.Weights * prevActivation;
                ColumnVector z12 = z1 + currentLayer.Biases;
                prevActivation = currentLayer.ActivationFunction.Activate(z12);
            }
            return prevActivation;
        }

        // Note: this is specialized for 2 layers (input, hidden, output). Great as a reference
        // But not generalized for many layers
        public void BackProp_2layer(TrainingPair trainingPair, ColumnVector predictedOut)
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
            ColumnVector ActivationPartial = outputLayer.ActivationFunction.Derivative();  // sigmoid partial derivative
            ColumnVector w2_sigma = LossPartial * ActivationPartial;

            // Remember that the weights in the weight matrix are ROWS ...
            // so the dot product of row1 and output vector or activation vector minus the bias is = Z (the input to the activation function)

            // so the gradient w' matrix needs to be rows of gradient weights (or weight deltas) that we get from all the partial derivative shenanigans
            Matrix scaledGradientWeights_outputLayer = this.TrainingRate * BuildGradientWeights(hiddenLayer.LastActivationOutput, w2_sigma);
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
            ColumnVector DOl_DZL = hiddenLayer.GetActivationFunctionDerivative() * sum_over_all_de_dOl;
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

        //
        // generalized back propagation for many layers
        //
        public void BackProp(TrainingPair trainingPair, ColumnVector predictedOut)
        {
            //
            // Special case the Output Layer
            //
            WeightedLayer outputLayer = WeightedLayers[LayerCount - 1];
            WeightedLayer secondToLastLayer = WeightedLayers[LayerCount - 2];

            ColumnVector outputLayerSigma;
            // special case softmax and cross entropy
            if (outputLayer.ActivationFunction is SoftMax)
            {
                Debug.Assert(this.LossFunction is CategoricalCrossEntropy ||
                             this.LossFunction is SparseCategoricalCrossEntropy ||
                             this.LossFunction is VanillaCrossEntropy);

                // after all the crazy derivatives of softmax * crossentrotpy, we just end up with: a - y
                // which is 'activtion' of softmax minus the truth vector.  must be onehot encoded
                outputLayerSigma = outputLayer.LastActivationOutput - trainingPair.Output;
            }
            else
            {
                // partial product, before we start the per-w differentials.
                ColumnVector LossPartial = this.LossFunction.Derivative(trainingPair.Output, predictedOut);
                ColumnVector ActivationPartial = outputLayer.ActivationFunction.Derivative();
                outputLayerSigma = LossPartial * ActivationPartial;
            }

            Matrix weightGradient_outputlayer = BuildGradientWeights(secondToLastLayer.LastActivationOutput, outputLayerSigma);
            ColumnVector biasGradient = outputLayerSigma * 1.0;

            outputLayer.LastSigma = outputLayerSigma;
            outputLayer.AccumulateGradients(weightGradient_outputlayer, biasGradient);

            //
            // Now do back prop through all the hidden layers (ie: not the output). Special case for the input layer. That's why we go all the way to zero.
            //
            for (int L = LayerCount - 2; L >= 0; L--)
            {
                WeightedLayer currentLayer = WeightedLayers[L];
                WeightedLayer layerToTheRight = WeightedLayers[L + 1];

                ColumnVector PrevLayerSigma = layerToTheRight.LastSigma;
                ColumnVector sum_over_all_de_dOl = layerToTheRight.Weights.GetTransposedMatrix() * PrevLayerSigma;
                ColumnVector DOl_DZL = currentLayer.GetActivationFunctionDerivative() * sum_over_all_de_dOl;
                currentLayer.LastSigma = DOl_DZL;

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
                    activationToTheLeft = WeightedLayers[L - 1].LastActivationOutput;
                }

                Matrix gradientWeights = BuildGradientWeights(activationToTheLeft, DOl_DZL);
                ColumnVector gradientBias = DOl_DZL;
                currentLayer.AccumulateGradients(gradientWeights, gradientBias);
            }
        }

        public void UpdateWeightsAndBiases()
        {
            // Now go through the network and update all weights and biases with the learning rate
            for (int L = LayerCount - 1; L >= 0; L--)
            {
                WeightedLayer layer = WeightedLayers[L];
                layer.UpdateWeightsAndBiases();
            }
        }

        private Matrix BuildGradientWeights(ColumnVector lastActivation, ColumnVector sigma)
        {
            // Do outer product
            // we want all the sigmas (on the right) times all the Outpus (from the left) to look like the wiehgt matrix
            // where the top row represents the weights of the entier (left) layer.
            // this matrix is the D(E)/D(w) final result, and the partial derivative of the massive dot product at each right node, per neft node is simply the output of the left node
            //sigma = this.TrainingRate * sigma;
            // Matrix scaledGradientWeights = sigma * lastActivation.Transpose();

            // temp: inefficient for debugging
            Matrix gradientDelta = sigma * lastActivation.Transpose(); // outer product:  o1s1 o2s1 o3s1 o4s1 ... onS1    o1s2 o2s2 o3s2 ... oNs2 
            return gradientDelta;
        }

        public double GetTotallLoss(TrainingPair tp, ColumnVector predicted)
        {
            ColumnVector lossVec = this.LossFunction.Error(tp.Output, predicted);
            return lossVec.Sum();
        }

        public ColumnVector GetLossVector(TrainingPair tp, ColumnVector predicted)
        {
            return this.LossFunction.Error(tp.Output, predicted);
        }

        public double GetAveragelLoss(TrainingPair tp, ColumnVector predicted)
        {
            return this.GetTotallLoss(tp, predicted) / (double)predicted.Size;
        }
    }
}
