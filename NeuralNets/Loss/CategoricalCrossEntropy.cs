using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    public class CategoricalCrossEntropy : ILossFunction
    {
        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            throw new NotImplementedException();
        }

        // not written for more than one sample.
        // Expects "predicted" to be the output from SoftMax
        public float ScalarLoss(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            // use sum once, since this is one dimensional. For batch processing it'll be a 2d matrix and we'll use OneHotEconde (below)
            return this.Error(truth, predicted).Sum();
        }

        public float ScalarLossBatch(MatrixBase truth, MatrixBase predicted)
        {
            MatrixBase oneHotSamples = this.OneHotEncode(truth);
            MatrixBase logYHAT = predicted.Log();
            MatrixBase YTimeYHat = oneHotSamples * logYHAT;
            float sum = YTimeYHat.Sum();
            return sum;
        }

        // Assumes number of "classes" is the same as sample.Size.
        // In other words, the sample vector (predicted) has X entries, let's say, and that's exactly the number of classes.
        public MatrixBase OneHotEncode(MatrixBase samples)
        {
            int N = samples.Rows;
            float invN = -1 / (float)N;

            MatrixBase m = MatrixFactory.CreateMatrix(samples.Rows, samples.Cols);
            for (int i = 0; i < N; i++)
            {
                //ColumnVectorBase sample = samples[i];
                for (int j = 0; j < samples.Cols; j++)
                {
                    if (samples[i, j] >= 0.99999)
                    {
                        m[i, j] = 1;
                        break; // assumes there's only one. might want to assert this?  TODO
                    }
                }
            }
            return m;
        }

        public ColumnVectorBase Error(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            return -1 * truth * predicted.Log();
        }

        public ColumnVectorBase Error(Tensor truth, Tensor predicted)
        {
            throw new NotImplementedException();
        }
    }
}