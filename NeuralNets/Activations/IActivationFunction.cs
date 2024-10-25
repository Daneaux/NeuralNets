using MatrixLibrary;

namespace NeuralNets
{
    public interface IActivationFunction
    {
        Tensor LastActivation { get; }
        AvxColumnVector Activate(AvxColumnVector input);
        AvxColumnVector Derivative(AvxColumnVector lastActivation);
        AvxMatrix Activate(AvxMatrix input);
        List<AvxMatrix> Activate(List<AvxMatrix> input);
    }
}