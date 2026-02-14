using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System;

namespace NeuralNets
{
    public class SparseCategoricalCrossEntropy : ILossFunction
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

        private static ColumnVectorBase CreateOneHot(int size, int classIndex)
        {
            float[] oneHotVec = new float[size];
            oneHotVec[classIndex] = 1;
            return MatrixFactory.CreateColumnVector(oneHotVec);
        }

        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            ColumnVectorBase softmaxPred = SoftMax(predicted);
            int classIndex = (int)truth[0];
            ColumnVectorBase oneHot = CreateOneHot(predicted.Size, classIndex);
            return softmaxPred - oneHot;
        }

        public ColumnVectorBase Error(Tensor truth, Tensor predicted)
        {
            throw new NotImplementedException();
        }

        public float ScalarLoss(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            ColumnVectorBase softmaxPred = SoftMax(predicted);
            int classIndex = (int)truth[0];
            float logProb = (float)Math.Log(softmaxPred[classIndex]);
            return -logProb;
        }

        public float ScalarLossBatch(MatrixBase truth, MatrixBase predicted)
        {
            MatrixBase softmaxPred = SoftMax(predicted);
            int batchSize = truth.Rows;
            int numClasses = predicted.Cols;

            float sum = 0;
            for (int i = 0; i < batchSize; i++)
            {
                int classIndex = (int)truth[i, 0];
                sum += (float)Math.Log(softmaxPred[i, classIndex]);
            }
            return -sum / batchSize;
        }

        ColumnVectorBase ILossFunction.Error(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            ColumnVectorBase softmaxPred = SoftMax(predicted);
            int classIndex = (int)truth[0];
            ColumnVectorBase oneHot = CreateOneHot(predicted.Size, classIndex);
            return -1 * oneHot * softmaxPred.Log();
        }
    }
}
