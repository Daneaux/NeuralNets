using System.Diagnostics;
using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    // Mean Squared Error Loss - matches PyTorch/TorchSharp MSELoss behavior
    // Loss = (predicted - actual)^2 (without the 0.5 factor)
    // Derivative = 2 * (predicted - actual)
    public class MeanSquaredErrorLoss : ILossFunction
    {
        public float Error(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            Debug.Assert(truth.Size == predicted.Size, "Truth and predicted vectors must be the same size.");
            var sq = (predicted - truth) * (predicted - truth);
            return sq.Sum() / truth.Size;
        }

        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted) => 2 * (predicted - truth);

        public float Error(Tensor truth, Tensor predicted)
        {
            var t = truth as AnnTensor;
            var p = predicted as AnnTensor;
            return Error(t.ColumnVector, p.ColumnVector);
        }
    }
}
