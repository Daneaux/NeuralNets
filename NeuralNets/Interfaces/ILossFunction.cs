﻿using System;

namespace NeuralNets
{
    /*
     * https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
     * 
     --- Regression Problem
A problem where you predict a real-value quantity.

Output Layer Configuration: One node with a linear activation unit.
Loss Function: Mean Squared Error (MSE).

---- Binary Classification Problem
A problem where you classify an example as belonging to one of two classes.

The problem is framed as predicting the likelihood of an example belonging to class one, e.g. the class that you assign the integer value 1, whereas the other class is assigned the value 0.

Output Layer Configuration: One node with a sigmoid activation unit.
Loss Function: Cross-Entropy, also referred to as Logarithmic loss.

----Multi-Class Classification Problem
A problem where you classify an example as belonging to one of more than two classes.

The problem is framed as predicting the likelihood of an example belonging to each class.

Output Layer Configuration: One node for each class using the softmax activation function.
Loss Function: Cross-Entropy, also referred to as Logarithmic loss.
*/


    public interface ILossFunction
    {
        ColumnVector Error(ColumnVector truth, ColumnVector predicted);
        ColumnVector Derivative(ColumnVector truth, ColumnVector predicted);
    }

    public class DeltaError : ILossFunction
    {
        public ColumnVector Derivative(ColumnVector truth, ColumnVector predicted)
        {
            throw new NotImplementedException();
        }

        public ColumnVector Error(ColumnVector truth, ColumnVector predicted) => predicted - truth;
    }

    // 1/2 * (precited - actual)^2
    public class SquaredLoss : ILossFunction
    {
        public ColumnVector Error(ColumnVector truth, ColumnVector predicted) => 0.5 * (predicted - truth) * (predicted - truth);

        public ColumnVector Derivative(ColumnVector truth, ColumnVector predicted) => (predicted - truth);
    }

    public class CategoricalCrossEntropy : ILossFunction
    {
        public ColumnVector Derivative(ColumnVector truth, ColumnVector predicted)
        {
            throw new NotImplementedException();
        }

        // not written for more than one sample.
        // Expects "predicted" to be the output from SoftMax
        public double ScalarLoss(ColumnVector truth, ColumnVector predicted)
        {
            // use sum once, since this is one dimensional. For batch processing it'll be a 2d matrix and we'll use OneHotEconde (below)
            return this.Error(truth, predicted).Sum();
        }

        public double ScalarLossBatch(Matrix truth, Matrix predicted)
        {
            Matrix oneHotSamples = this.OneHotEncode(truth);
            Matrix logYHAT = predicted.Log();
            Matrix YTimeYHat = oneHotSamples * logYHAT;
            double sum = YTimeYHat.Sum();
            return sum;
        }

        // Assumes number of "classes" is the same as sample.Size.
        // In other words, the sample vector (predicted) has X entries, let's say, and that's exactly the number of classes.
        public Matrix OneHotEncode(Matrix samples)
        {
            int N = samples.Rows;
            double invN = -1.0 / (double)N;

            Matrix m = new Matrix(samples.Rows, samples.Cols);
            for (int i = 0; i < N; i++)
            {
                //ColumnVector sample = samples[i];
                for (int j = 0; j < samples.Cols; j++)
                {
                    if (samples[i, j] >= 0.99999)
                    {
                        m[i, j] = 1.0;
                        break; // assumes there's only one. might want to assert this?  TODO
                    }
                }
            }
            return m;
        }

        public ColumnVector Error(ColumnVector truth, ColumnVector predicted)
        {
            return -1.0 * truth * predicted.Log();
        }
    }

    // Cross Entropy ==  -Sum [ truth * log(predicted) ]
    public class VanillaCrossEntropy : ILossFunction
    {
        public ColumnVector Error(ColumnVector truth, ColumnVector predicted)
        {
            ColumnVector v1 = -1 * truth * predicted.Log();
            return v1;
        }

        public ColumnVector Derivative(ColumnVector truth, ColumnVector predicted)
        {
            throw new NotImplementedException();
        }
    }


    public class SparseCategoricalCrossEntropy : ILossFunction
    {
        public ColumnVector Derivative(ColumnVector truth, ColumnVector predicted)
        {
            throw new NotImplementedException();
        }

        // not written for more than one sample.
        // Expects "predicted" to be the output from SoftMax
        public double ScalarLoss(ColumnVector truth, ColumnVector predicted)
        {
            double NInv = -1.0;
            ColumnVector logYHat = predicted.Log();
            return NInv * logYHat.Sum();
        }

        // not clear if the input needs to be an array with the actual value of the class, like [0,0,0,4,0,0,0,0] which means the 4th index is a four, that is we recognized a four in the image input.
        // totally broken, need to re-read this and get it right:
        // https://arjun-sarkar786.medium.com/implementation-of-all-loss-functions-deep-learning-in-numpy-tensorflow-and-pytorch-e20e72626ebd
        public double ScalarLossBatch(Matrix truth, Matrix predicted)
        {
            throw new NotImplementedException();
            int N = truth.Rows;
            double invN = -1.0 / (double)N;

            Matrix logYHAT = predicted.Log();
            Matrix YTimeYHat = 1.0 * logYHAT;
            double sum = YTimeYHat.Sum();
            return sum;
        }

        ColumnVector ILossFunction.Error(ColumnVector truth, ColumnVector predicted)
        {
            throw new NotImplementedException();
        }
    }
}