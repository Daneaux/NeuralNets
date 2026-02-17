using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MatrixLibrary.BaseClasses;

namespace NeuralNets.Network
{
    internal class Appendix_reference
    {

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


        // for reference
#if false

        private void SetLayerGradients(int L, MatrixBase weightGradient, ColumnVectorBase biasGradient)
        {
            this.BiasGradient[L] = biasGradient;
            this.WeightGradient[L] = weightGradient;
        }

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
    }
}
