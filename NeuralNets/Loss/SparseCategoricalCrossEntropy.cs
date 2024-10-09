using MatrixLibrary;

namespace NeuralNets
{
    public class SparseCategoricalCrossEntropy : ILossFunction
    {
        public AvxColumnVector Derivative(AvxColumnVector truth, AvxColumnVector predicted)
        {
            throw new NotImplementedException();
        }

        public AvxColumnVector Error(Tensor truth, Tensor predicted)
        {
            throw new NotImplementedException();
        }

        // not written for more than one sample.
        // Expects "predicted" to be the output from SoftMax
        public float ScalarLoss(AvxColumnVector truth, AvxColumnVector predicted)
        {
            float NInv = -1;
            AvxColumnVector logYHat = predicted.Log();
            return NInv * logYHat.Sum();
        }

        // not clear if the input needs to be an array with the actual value of the class, like [0,0,0,4,0,0,0,0] which means the 4th index is a four, that is we recognized a four in the image input.
        // totally broken, need to re-read this and get it right:
        // https://arjun-sarkar786.medium.com/implementation-of-all-loss-functions-deep-learning-in-numpy-tensorflow-and-pytorch-e20e72626ebd
        public float ScalarLossBatch(AvxMatrix truth, AvxMatrix predicted)
        {
            throw new NotImplementedException();
            int N = truth.Rows;
            float invN = -1 / (float)N;

            AvxMatrix logYHAT = predicted.Log();
            AvxMatrix YTimeYHat = 1 * logYHAT;
            float sum = YTimeYHat.Sum();
            return sum;
        }

        AvxColumnVector ILossFunction.Error(AvxColumnVector truth, AvxColumnVector predicted)
        {
            throw new NotImplementedException();
        }
    }
}