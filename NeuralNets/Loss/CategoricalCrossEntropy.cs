using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System;

namespace NeuralNets
{
    public class CategoricalCrossEntropy : ILossFunction
    {
        private static ColumnVectorBase SoftMax(ColumnVectorBase input)
        {
            float max = input.GetMax();
            float scaleFactor = 0;
            float[] softMaxVec = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
            {
                scaleFactor += (float)Math.Exp(input[i] - max);
            }
            for (int i = 0; i < input.Size; i++)
            {
                softMaxVec[i] = (float)(Math.Exp(input[i] - max) / scaleFactor);
            }
            return MatrixFactory.CreateColumnVector(softMaxVec);
        }

        private static MatrixBase SoftMax(MatrixBase input)
        {
            int rows = input.Rows;
            int cols = input.Cols;
            float[,] result = new float[rows, cols];

            for (int r = 0; r < rows; r++)
            {
                float max = float.MinValue;
                for (int c = 0; c < cols; c++)
                {
                    if (input[r, c] > max) max = input[r, c];
                }

                float scaleFactor = 0;
                for (int c = 0; c < cols; c++)
                {
                    scaleFactor += (float)Math.Exp(input[r, c] - max);
                }

                for (int c = 0; c < cols; c++)
                {
                    result[r, c] = (float)(Math.Exp(input[r, c] - max) / scaleFactor);
                }
            }

            return MatrixFactory.CreateMatrix(result);
        }

        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            ColumnVectorBase softmaxPred = SoftMax(predicted);
            return softmaxPred - truth;
        }

        public float ScalarLoss(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            ColumnVectorBase softmaxPred = SoftMax(predicted);
            ColumnVectorBase logSoftmax = softmaxPred.Log();
            ColumnVectorBase result = truth * logSoftmax;
            return -result.Sum();
        }

        public float ScalarLossBatch(MatrixBase truth, MatrixBase predicted)
        {
            MatrixBase softmaxPred = SoftMax(predicted);
            MatrixBase oneHotSamples = OneHotEncode(truth);
            MatrixBase logYHAT = softmaxPred.Log();

            int batchSize = truth.Rows;
            int numClasses = truth.Cols;
            float sum = 0;
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < numClasses; j++)
                {
                    sum += oneHotSamples[i, j] * logYHAT[i, j];
                }
            }
            return -sum;
        }

        public MatrixBase OneHotEncode(MatrixBase samples)
        {
            int N = samples.Rows;

            MatrixBase m = MatrixFactory.CreateMatrix(samples.Rows, samples.Cols);
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < samples.Cols; j++)
                {
                    if (samples[i, j] >= 0.99999)
                    {
                        m[i, j] = 1;
                        break;
                    }
                }
            }
            return m;
        }

        public ColumnVectorBase Error(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            ColumnVectorBase softmaxPred = SoftMax(predicted);
            ColumnVectorBase logSoftmax = softmaxPred.Log();
            return -1 * truth * logSoftmax;
        }

        public ColumnVectorBase Error(Tensor truth, Tensor predicted)
        {
            var annTruth = truth as AnnTensor;
            var annPred = predicted as AnnTensor;

            if (annTruth?.ColumnVector != null && annPred?.ColumnVector != null)
            {
                return Error(annTruth.ColumnVector, annPred.ColumnVector);
            }

            if (annTruth?.Matrix != null && annPred?.Matrix != null)
            {
                MatrixBase softmaxPred = SoftMax(annPred.Matrix);
                MatrixBase logSoftmax = softmaxPred.Log();
                MatrixBase error = annTruth.Matrix.Multiply(logSoftmax);
                return MatrixHelpers.UnrollMatricesToColumnVector(new List<MatrixBase> { error.Multiply(-1f) });
            }

            throw new NotSupportedException("Only AnnTensor is supported for Tensor error calculation");
        }
    }
}
