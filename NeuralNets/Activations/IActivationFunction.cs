using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    public interface IActivationFunction
    {
        Tensor LastActivation { get; }
        ColumnVectorBase Activate(ColumnVectorBase input);
        ColumnVectorBase Derivative(ColumnVectorBase lastActivation);
        MatrixBase Activate(MatrixBase input);
        List<MatrixBase> Activate(List<MatrixBase> input);
    }
}