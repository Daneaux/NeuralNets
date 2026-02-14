using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    public class SparseCategoricalCrossEntropy : ILossFunction
    {
        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            throw new NotImplementedException();
        }

        public ColumnVectorBase Error(Tensor truth, Tensor predicted)
        {
            throw new NotImplementedException();
        }

        // not written for more than one sample.
        // Expects "predicted" to be the output from SoftMax
        public float ScalarLoss(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            float NInv = -1;
            ColumnVectorBase logYHat = predicted.Log();
            return NInv * logYHat.Sum();
        }

        // not clear if the input needs to be an array with the actual value of the class, like [0,0,0,4,0,0,0,0] which means the 4th index is a four, that is we recognized a four in the image input.
        // totally broken, need to re-read this and get it right:
        // https://arjun-sarkar786.medium.com/implementation-of-all-loss-functions-deep-learning-in-numpy-tensorflow-and-pytorch-e20e72626ebd
        public float ScalarLossBatch(MatrixBase truth, MatrixBase predicted)
        {
            throw new NotImplementedException();
            // todo: this is probably correct, but currently unused, not sure why.
            int N = truth.Rows;
            float invN = -1 / (float)N;

            MatrixBase logYHAT = predicted.Log();
            MatrixBase YTimeYHat = 1 * logYHAT;
            float sum = YTimeYHat.Sum();
            return sum;
        }

        ColumnVectorBase ILossFunction.Error(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            throw new NotImplementedException();
        }
    }
}