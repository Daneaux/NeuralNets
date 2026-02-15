using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    // 1/2 * (predicted - actual)^2
    public class SquaredLoss : ILossFunction
    {
        public float Error(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            var lossVec = 0.5F * (predicted - truth) * (predicted - truth);
            return lossVec.Sum();
        }

        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted) => (predicted - truth);

        public float Error(Tensor truth, Tensor predicted)
        {
            var t = truth as AnnTensor;
            var p = predicted as AnnTensor;
            return Error(t.ColumnVector, p.ColumnVector);
        }
    }
}