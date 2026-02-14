using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System;

namespace NeuralNets
{
    public class VanillaCrossEntropy : ILossFunction
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

        public ColumnVectorBase Error(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            ColumnVectorBase softmaxPred = SoftMax(predicted);
            ColumnVectorBase v1 = -1 * truth * softmaxPred.Log();
            return v1;
        }

        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            ColumnVectorBase softmaxPred = SoftMax(predicted);
            return softmaxPred - truth;
        }

        public ColumnVectorBase Error(Tensor truth, Tensor predicted)
        {
            throw new NotImplementedException();
        }
    }
}
