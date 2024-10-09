using MatrixLibrary;

namespace NeuralNets
{
    // 1/2 * (precited - actual)^2
    public class SquaredLoss : ILossFunction
    {
        public AvxColumnVector Error(AvxColumnVector truth, AvxColumnVector predicted) => 0.5F * (predicted - truth) * (predicted - truth);

        public AvxColumnVector Derivative(AvxColumnVector truth, AvxColumnVector predicted) => (predicted - truth);

        public AvxColumnVector Error(Tensor truth, Tensor predicted)
        {
            var t = truth as AnnTensor;
            var p = predicted as AnnTensor;
            return Error(t.ColumnVector, p.ColumnVector);
        }
    }
}