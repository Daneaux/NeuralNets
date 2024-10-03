using MatrixLibrary;

namespace NeuralNets
{
    public abstract class Layer
    {
        public int IncomingDataPoints { get; }
        public int NumNodes { get; private set; }
        public IActivationFunction ActivationFunction { get; private set; }

        public abstract Tensor FeedFoward(Tensor input);

        protected Layer(
            int nodeCount, 
            IActivationFunction activationFunction, 
            int incomingDataPoints, 
            int randomSeed = 12341324)
        {
            this.NumNodes = nodeCount;
            ActivationFunction = activationFunction;
            IncomingDataPoints = incomingDataPoints;
        }

        public abstract void UpdateWeightsAndBiasesWithScaledGradients(Tensor weightGradient, Tensor biasGradient);
    }
}
