using NeuralNets;
using System.Diagnostics;

namespace NumReaderNetwork
{
    public class SoftMax : IActivationFunction
    {
        public ColumnVector? LastActivation { get; private set; }

        public ColumnVector Activate(ColumnVector input)
        {
            this.LastActivation = input.SoftmaxHelper();
            return this.LastActivation;
        }

        // TODO: this isn't trivial. need to re-read 
        // https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
        // https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
        public ColumnVector Derivative()
        {
            Debug.Assert(this.LastActivation != null);
            throw new InvalidOperationException("Don't call derivative on softmax, just use the softmax*crossentropy derivative which is a-y'");            
        }
    }
}