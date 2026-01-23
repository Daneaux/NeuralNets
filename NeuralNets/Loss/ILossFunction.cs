using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    /*
     * https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
     */
    public interface ILossFunction
    {
        ColumnVectorBase Error(ColumnVectorBase truth, ColumnVectorBase predicted);
        ColumnVectorBase Error(Tensor truth, Tensor predicted);
        ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted);
    }
}