using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    // 1/2 * (precited - actual)^2
    public class SquaredLoss : ILossFunction
    {
        public ColumnVectorBase Error(ColumnVectorBase truth, ColumnVectorBase predicted) => 0.5F * (predicted - truth) * (predicted - truth);

        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted) => (predicted - truth);

        public ColumnVectorBase Error(Tensor truth, Tensor predicted)
        {
            var t = truth as AnnTensor;
            var p = predicted as AnnTensor;
            return Error(t.ColumnVector, p.ColumnVector);
        }
    }
}