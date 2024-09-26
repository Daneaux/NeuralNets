using MatrixLibrary;
using System.Diagnostics;

namespace NeuralNets
{
    public class SoftMax : IActivationFunction
    {
        public ColumnVector Activate(ColumnVector input)
        {
            return input.SoftmaxHelper();
        }

        public AvxColumnVector Activate(AvxColumnVector input)
        {
            throw new NotImplementedException();
        }

        public AvxMatrix Activate(AvxMatrix input)
        {
            throw new NotImplementedException();
        }

        // TODO: this isn't trivial. need to re-read 
        // https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
        // https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
        public ColumnVector Derivative(ColumnVector lastActivation)
        {
            Debug.Assert(lastActivation != null);
            throw new InvalidOperationException("Don't call derivative on softmax, just use the softmax*crossentropy derivative which is a-y'");
        }

        public AvxColumnVector Derivative(AvxColumnVector lastActivation)
        {
            throw new InvalidOperationException("Don't call derivative on softmax, just use the softmax*crossentropy derivative which is a-y'");
        }
    }
}