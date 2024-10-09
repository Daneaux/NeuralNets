using MatrixLibrary;

namespace NeuralNets
{
    /*
     * https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
     */
    public interface ILossFunction
    {
        AvxColumnVector Error(AvxColumnVector truth, AvxColumnVector predicted);
        AvxColumnVector Error(Tensor truth, Tensor predicted);
        AvxColumnVector Derivative(AvxColumnVector truth, AvxColumnVector predicted);
    }
}