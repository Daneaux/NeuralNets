using Microsoft.Win32.SafeHandles;
using NeuralNets;
using System;
using System.Diagnostics;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace NumReaderNetwork
{
    public class NumberReaderANN
    {
        public double TrainingRate { get; }
        public int InputDim { get; private set; }
        public int OutputLayerDim { get; private set; }
        //public InputLayer InputLayer { get; private set; }
        public List<WeightedLayer> WeightedLayers { get;  set; } = [];
        public ILossFunction LossFunction { get; private set; }
        public WeightedLayer OutputLayer { get { return WeightedLayers[WeightedLayers.Count - 1]; } }

        public NumberReaderANN(int inputDim, int hiddenLayerDim, int outputDim)
        {
            this.TrainingRate = 0.1;
            this.InputDim = inputDim;
            this.OutputLayerDim = outputDim;
            this.LossFunction = new SquaredLoss();
            WeightedLayers.Add(new WeightedLayer(hiddenLayerDim, new ReLUActivaction(), InputDim));
            WeightedLayers.Add(new WeightedLayer(OutputLayerDim, new SigmoidActivation(), hiddenLayerDim));
        }

        public NumberReaderANN(int inputDim, List<WeightedLayer> layers, double trainingRate, ILossFunction lossFunction)
        {
            Debug.Assert(layers != null);
            Debug.Assert(layers.Count > 0);
            if(lossFunction == null)
            {
               lossFunction = new SquaredLoss();
            }
            this.InputDim = inputDim;
            this.TrainingRate = trainingRate;
            this.LossFunction = lossFunction;
            this.WeightedLayers = layers;
            this.OutputLayerDim = this.OutputLayer.NumNodes;
        }

        public void TrainWithImages(int iterations)
        {
            int i = 0;
            foreach (Image image in MnistReader.ReadTrainingData())
            {
                ColumnVector inputVector = ImageDataToColumnVector(image);
                ColumnVector outputVector = LabelDataToColumnVector(image);
                TrainingPair trainingPair = new TrainingPair(inputVector, outputVector);

                ColumnVector prediction = this.FeedForward(inputVector);
                BackProp(trainingPair, prediction);

                double error = GetAveragelLoss(trainingPair, prediction);
                if (i % 100 == 0)
                {
                    Console.WriteLine($"{i}: Loss = {error}\n");
                }

                if(i > iterations)
                {
                    break;
                }
                i++;
            }
        }

        private ColumnVector LabelDataToColumnVector(Image image)
        {
            // convert the label data (0,1,2, ...) into a columnvector. if the label is 7 (ie: byte == 7), then set the 7th double to 1.0
            double[] labelData = new double[this.OutputLayerDim];
            labelData[(int)image.Label] = 1.0;
            return new ColumnVector(labelData);
        }

        private static ColumnVector ImageDataToColumnVector(Image image)
        {
            // convert the image data into a columnvector
            double[] imageData = new double[image.Size];
            int i = 0;
            foreach (byte b in image.Data)
            {
                imageData[i++] = (double)b;
            }

            return new ColumnVector(imageData);
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
                currentLayer.SetActivationOutput(o1);
                prevActivation = o1;
            }
            return prevActivation;
        }

        // TODO: this works for two layers. Obviously needs to be generalized to N layers.
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
            ColumnVector LossPartial = this.LossFunction.Derivative(trainingPair.Output, predictedOut);
            ColumnVector ActivationPartial = outputLayer.ActivationFunction.Derivative();  // sigmoid partial derivative
            ColumnVector w2_sigma = LossPartial * ActivationPartial;

            // Remember that the weights in the weight matrix are ROWS ...
            // so the dot product of row1 and output vector or activation vector minus the bias is = Z (the input to the activation function)

            // so the gradient w' matrix needs to be rows of gradient weights (or weight deltas) that we get from all the partial derivative shenanigans
            Matrix scaledGradientWeights_outputLayer = BuildScaledGradientWeights(hiddenLayer.ActivationFunction.LastActivation, w2_sigma);
            ColumnVector b2_delta = this.TrainingRate * w2_sigma * 1.0;  // d(z2)/b  .. i think?


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
            outputLayer.UpdateWeights(scaledGradientWeights_outputLayer);
            outputLayer.UpdateBiases(b2_delta);
            hiddenLayer.UpdateWeights(scaledGradientWeights_hiddenLayer);
            hiddenLayer.UpdateBiases(b1_delta);
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
