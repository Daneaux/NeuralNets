using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    // Mean Squared Error Loss - matches PyTorch/TorchSharp MSELoss behavior
    // Loss = (predicted - actual)^2 (without the 0.5 factor)
    // Derivative = 2 * (predicted - actual)
    public class MeanSquaredErrorLoss : ILossFunction
    {
        public ColumnVectorBase Error(ColumnVectorBase truth, ColumnVectorBase predicted) => (predicted - truth) * (predicted - truth);

        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted) => 2 * (predicted - truth);

        public ColumnVectorBase Error(Tensor truth, Tensor predicted)
        {
            var t = truth as AnnTensor;
            var p = predicted as AnnTensor;
            return Error(t.ColumnVector, p.ColumnVector);
        }
    }
}
