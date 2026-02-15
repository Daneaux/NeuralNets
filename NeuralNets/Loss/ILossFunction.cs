using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    /*
     * https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
     */
    public interface ILossFunction
    {
        float Error(ColumnVectorBase truth, ColumnVectorBase predicted);
        float Error(Tensor truth, Tensor predicted);
        ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted);
    }
}