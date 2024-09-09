using NeuralNets;
using System.Diagnostics;

namespace NeuralNets
{
    public class SoftMax : IActivationFunction
    {

        public ColumnVector Activate(ColumnVector input)
        {
            return input.SoftmaxHelper();
        }

        // TODO: this isn't trivial. need to re-read 
        // https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
        // https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
        public ColumnVector Derivative(ColumnVector lastActivation)
        {
            Debug.Assert(lastActivation != null);
            throw new InvalidOperationException("Don't call derivative on softmax, just use the softmax*crossentropy derivative which is a-y'");            
        }
    }
}