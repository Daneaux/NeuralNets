using MatrixLibrary;

namespace NeuralNets
{
    public abstract class Layer
    {
        public InputOutputShape InputShape { get; }
        public abstract InputOutputShape OutputShape { get; }
        public int NumNodes { get; private set; }
        public IActivationFunction ActivationFunction { get; private set; }

        public abstract Tensor FeedFoward(Tensor input);

        protected Layer(
            InputOutputShape inputShape,
            int nodeCount, 
            IActivationFunction activationFunction, 
            int randomSeed = 12341324)
        {
            NumNodes = nodeCount;
            InputShape = inputShape;
            ActivationFunction = activationFunction;
        }

        public abstract void UpdateWeightsAndBiasesWithScaledGradients(Tensor weightGradient, Tensor biasGradient);
    }
}
