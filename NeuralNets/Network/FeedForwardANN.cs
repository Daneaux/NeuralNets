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
        public int InputDim { get; protected set; }
        public int OutputLayerDim { get; protected set; }
        //public InputLayer InputLayer { get; private set; }
        public List<WeightedLayer> WeightedLayers { get; set; } = [];
        public ILossFunction LossFunction { get; protected set; }
        public WeightedLayer OutputLayer { get { return WeightedLayers[WeightedLayers.Count - 1]; } }

        protected GeneralFeedForwardANN(int inputDim, double trainingRate)
        {
            this.InputDim = inputDim;
            this.TrainingRate = trainingRate;
        }

        public GeneralFeedForwardANN(int inputDim, List<WeightedLayer> layers, double trainingRate, ILossFunction lossFunction)
        {
            Debug.Assert(layers != null);
            Debug.Assert(layers.Count > 0);
            if (lossFunction == null)
            {
                lossFunction = new SquaredLoss();
            }
            this.InputDim = inputDim;
            this.TrainingRate = trainingRate;
            this.LossFunction = lossFunction;
            this.WeightedLayers = layers;
            this.OutputLayerDim = this.OutputLayer.NumNodes;
        }

        public ColumnVector FeedForward(ColumnVector inputVec)
        {
            Debug.Assert(inputVec.Size == this.InputDim);
            Debug.Assert(WeightedLayers.Count == 2);
            ColumnVector prevActivation = inputVec;
            for (int i = 0; i < 2; i++)
            {
                WeightedLayer currentLayer = WeightedLayers[i];
                ColumnVector z1 = currentLayer.Weights * prevActivation;
                ColumnVector z12 = z1 + currentLayer.Biases;
                ColumnVector o1 = currentLayer.ActivationFunction.Activate(z12);
                prevActivation = o1;
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
            Matrix scaledGradientWeights_outputLayer = BuildScaledGradientWeights(hiddenLayer.LastActivationOutput, w2_sigma);
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
            Matrix scaledGradientWeights_hiddenLayer = BuildScaledGradientWeights(trainingPair.Input, DOl_DZL);

            // partial biases full equation:
            // D(E)/D(bb) = D(zl)/D(bb) * d(Ol)/d(zl) * SUM_OVER_ALL_OUTGOING_EDGES[ D(E)/D(Ol) ]   (for example de1/dzl + de0/dzl + de2/dzl ... deN/dzl)
            // Note all the terms are the same except the first : dzl/dbb
            ColumnVector b1_delta = this.TrainingRate * DOl_DZL * 1.0;

            // UPDATE THESE WEIGHTS AFTER BACK PROP IS DONE
            // Now: Update W2 weight matrix with w2_delta (and same for b)
            outputLayer.StashDeltas(scaledGradientWeights_outputLayer, b2_delta);
            outputLayer.UpdateWeights();
            outputLayer.UpdateBiases();

            hiddenLayer.StashDeltas(scaledGradientWeights_hiddenLayer, b1_delta);
            hiddenLayer.UpdateWeights();
            hiddenLayer.UpdateBiases();
        }


        //
        // generalized back propagation for many layers
        //
        public void BackProp(TrainingPair trainingPair, ColumnVector predictedOut)
        {
            int totalLayers = WeightedLayers.Count;

            //
            // Special case the Outpu Layer
            //
            WeightedLayer outputLayer = WeightedLayers[totalLayers - 1];
            WeightedLayer secondToLastLayer = WeightedLayers[totalLayers - 2];

            // partial product, before we start the per-w differentials.
            ColumnVector LossPartial = this.LossFunction.Derivative(trainingPair.Output, predictedOut);
            ColumnVector ActivationPartial = outputLayer.ActivationFunction.Derivative();
            ColumnVector outputLayeyrSigma = LossPartial * ActivationPartial;

            Matrix scaledGradientWeights_outputLayer = BuildScaledGradientWeights(secondToLastLayer.LastActivationOutput, outputLayeyrSigma);
            ColumnVector b2_delta = this.TrainingRate * outputLayeyrSigma * 1.0;

            outputLayer.LastSigma = outputLayeyrSigma;
            outputLayer.ScaledBiasDelta = b2_delta;
            outputLayer.ScaledWeightDelta = scaledGradientWeights_outputLayer;

            //
            // Now do back prop through all the hidden layers (ie: not the output). Special case for the input layer. That's why we go all the way to zero.
            //
            for (int L = totalLayers - 2; L >= 0; L--)
            {
                WeightedLayer currentLayer = WeightedLayers[L];
                WeightedLayer layerToTheRight = WeightedLayers[L + 1];

                ColumnVector PrevLayerSigma = layerToTheRight.LastSigma;
                ColumnVector sum_over_all_de_dOl = layerToTheRight.Weights.GetTransposedMatrix() * PrevLayerSigma;
                ColumnVector DOl_DZL = currentLayer.GetActivationFunctionDerivative() * sum_over_all_de_dOl;

                // If we're the first hiddne layer, L=0, then the last activation is the input from the input layer
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

                Matrix scaledGradientWeights = BuildScaledGradientWeights(activationToTheLeft, DOl_DZL);
                ColumnVector scaledBiasDelta = this.TrainingRate * DOl_DZL;
                currentLayer.StashDeltas(scaledGradientWeights, scaledBiasDelta);
            }

            // Now go through the network and update all weights and biases with the learning rate
            for (int L = totalLayers - 1; L >= 0; L--)
            {
                WeightedLayer layer = WeightedLayers[L];
                layer.UpdateWeights();
                layer.UpdateBiases();
            }
        }

        private Matrix BuildScaledGradientWeights(ColumnVector lastActivation, ColumnVector sigma)
        {
            // Do outer product
            // we want all the sigmas (on the right) times all the Outpus (from the left) to look like the wiehgt matrix
            // where the top row represents the weights of the entier (left) layer.
            // this matrix is the D(E)/D(w) final result, and the partial derivative of the massive dot product at each right node, per neft node is simply the output of the left node
            //sigma = this.TrainingRate * sigma;
            // Matrix scaledGradientWeights = sigma * lastActivation.Transpose();

            // temp: inefficient for debugging
            Matrix gradientDelta = sigma * lastActivation.Transpose(); // outer product:  o1s1 o2s1 o3s1 o4s1 ... onS1    o1s2 o2s2 o3s2 ... oNs2 
            // now scale by the learning rate
            Matrix scaledGradientWeights = this.TrainingRate * gradientDelta;  // inneficient, scale one of the vectors first.

            return scaledGradientWeights;
        }

        public double GetTotallLoss(TrainingPair tp, ColumnVector predicted)
        {
            ColumnVector lossVec = this.LossFunction.Error(tp.Output, predicted);
            return lossVec.ScalarSum();
        }

        public double GetAveragelLoss(TrainingPair tp, ColumnVector predicted)
        {
            return this.GetTotallLoss(tp, predicted) / (double)predicted.Size;
        }
    }
}
