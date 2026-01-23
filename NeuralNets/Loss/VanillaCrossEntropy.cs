using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    // Cross Entropy ==  -Sum [ truth * log(predicted) ]
    public class VanillaCrossEntropy : ILossFunction
    {
        public ColumnVectorBase Error(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            ColumnVectorBase v1 = -1 * truth * predicted.Log();
            return v1;
        }

        public ColumnVectorBase Derivative(ColumnVectorBase truth, ColumnVectorBase predicted)
        {
            throw new NotImplementedException();
        }

        public ColumnVectorBase Error(Tensor truth, Tensor predicted)
        {
            throw new NotImplementedException();
        }
    }
}