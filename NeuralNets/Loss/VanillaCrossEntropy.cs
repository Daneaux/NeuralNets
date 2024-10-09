using MatrixLibrary;

namespace NeuralNets
{
    // Cross Entropy ==  -Sum [ truth * log(predicted) ]
    public class VanillaCrossEntropy : ILossFunction
    {
        public AvxColumnVector Error(AvxColumnVector truth, AvxColumnVector predicted)
        {
            AvxColumnVector v1 = -1 * truth * predicted.Log();
            return v1;
        }

        public AvxColumnVector Derivative(AvxColumnVector truth, AvxColumnVector predicted)
        {
            throw new NotImplementedException();
        }

        public AvxColumnVector Error(Tensor truth, Tensor predicted)
        {
            throw new NotImplementedException();
        }
    }
}