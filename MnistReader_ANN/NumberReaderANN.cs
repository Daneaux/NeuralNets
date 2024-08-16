using Microsoft.Win32.SafeHandles;
using NeuralNets;
using System.Diagnostics;

namespace NumReaderNetwork
{
    public class NumberReaderANN
    {
        public double TrainingRate { get; }
        public int InputDim { get; private set; }
        public int HiddenLayerDim { get; private set; }
        public int OutputLayerDim { get; private set; }
        //public InputLayer InputLayer { get; private set; }
        public List<WeightedLayer> WeightedLayers { get;  set; } = [];
        public ILossFunction LossFunction { get; private set; }

        public NumberReaderANN(int inputDim, int hiddenLayerDim, int outputDim)
        {
            this.TrainingRate = 0.1;
            this.InputDim = inputDim;
            this.OutputLayerDim = outputDim;
            this.HiddenLayerDim = hiddenLayerDim;
            this.LossFunction = new SquaredLoss();
            this.InitializeNetwork();
        }

        public void InitializeNetwork()
        {
            WeightedLayers.Add(new WeightedLayer(HiddenLayerDim, new ReLUActivaction(), InputDim));
            WeightedLayers.Add(new WeightedLayer(OutputLayerDim, new SigmoidActivation(), HiddenLayerDim));
        }

        public ColumnVector FeedForward(ColumnVector inputVec)
        {
            Debug.Assert(inputVec.Size == this.InputDim);
            Debug.Assert(WeightedLayers.Count == 2);
            ColumnVector currentActivation = inputVec;
            //InputLayer = new InputLayer(currentActivation); // doesn't see necessary at all?
            for (int i = 0; i < 2; i++)
            {
                WeightedLayer currentLayer = WeightedLayers[i];
                ColumnVector z1 = currentLayer.Weights * currentActivation;
                ColumnVector z12 = z1 - currentLayer.Biases;
                ColumnVector o1 = currentLayer.ActivationFunction.Activate(z12);
                currentLayer.SetActivationOutput(o1);
                currentActivation = o1;
            }
            return currentActivation;
        }

        public void BackProp(TrainingPair trainingPair, ColumnVector predictedOut)
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
            ColumnVector LossPartial = this.LossFunction.Derivative(predictedOut, trainingPair.Output);                               // pred - actual
            ColumnVector ActivationPartial = outputLayer.ActivationFunction.Derivative();  // sigmoid partial derivative
            ColumnVector w2_sigma = LossPartial * ActivationPartial;

            // Remember that the weights in the weight matrix are ROWS ...
            // so the dot product of row1 and output vector or activation vector minus the bias is = Z (the input to the activation function)

            // so the gradient w' matrix needs to be rows of gradient weights (or weight deltas) that we get from all the partial derivative shenanigans
            Matrix scaledGradientWeights = BuildScaledGradientWeights(hiddenLayer.ActivationFunction.LastActivation, w2_sigma);
            ColumnVector b2_delta = this.TrainingRate * w2_sigma * 1.0;  // d(z2)/b  .. i think?


            // ----
            // For hidden layer:
            // v = the weights before the hiddne layer.
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
            ColumnVector de_dOl = outputLayer.Weights * w2_sigma;
            // NOTE: each entry of this column vector as the SUM_OVER_ALL_OUTGOING_EDGES for each HiddenLayer node.
            // for node 3, de_dOl[2] == the sum of all outgoing edges partial derivatives

           // ColumnVector DZl_Dv_times_dOl_dZl = trainingPair.Input * hiddenLayer.GetActivationFunctionDerivative();
            ColumnVector HiddenLayerWeightsGradient = hiddenLayer.GetActivationFunctionDerivative() * de_dOl;
            Matrix scaledGradientWeights_hiddenLayer = BuildScaledGradientWeights(trainingPair.Input, HiddenLayerWeightsGradient);
            ColumnVector b1_delta = this.TrainingRate * w2_sigma * 1.0;  // TODO: bug this is wrong, needs to be sigma of the hidden layer, not the output layer

            // UPDATE THESE WEIGHTS AFTER BACK PROP IS DONE
            // Now: Update W2 weight matrix with w2_delta (and same for b)
            outputLayer.UpdateWeights(scaledGradientWeights);
            outputLayer.UpdateBiases(b2_delta);
            hiddenLayer.UpdateWeights(scaledGradientWeights_hiddenLayer);
            hiddenLayer.UpdateBiases(b1_delta);
        }

        private Matrix BuildScaledGradientWeights(ColumnVector lastActivation, ColumnVector w2_sigma)
        {
            int numCols = lastActivation.Size;
            ColumnVector[] w2_delta = new ColumnVector[numCols];
            for (int i = 0; i < numCols; i++)
            {
                double o2 = lastActivation[i];
                w2_delta[i] = this.TrainingRate * w2_sigma * o2;                                // d(z2)/w
            }

            // Now turn that array of columns into a matrix and transpose
            Matrix scaledGradientWeights = new Matrix(w2_delta, w2_delta[0].Size);
            scaledGradientWeights = scaledGradientWeights.GetTransposedMatrix();
            return scaledGradientWeights;
        }

        public double GetAveragelLoss(TrainingPair tp, ColumnVector predicted)
        {
            ColumnVector lossVec = this.LossFunction.Error(tp.Output, predicted);
            return lossVec.ScalarSum() / (double)predicted.Size;
        }
    }
}
