using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    public abstract class Layer
    {
        public int NumNodes { get; private set; }
        public int RandomSeed { get; }
        public InputOutputShape InputShape { get; }
        public abstract InputOutputShape OutputShape { get; }
        protected readonly object GradientLock = new object();

        public MatrixBase LastWeightGradient { get; protected set; }
        public ColumnVectorBase LastBiasGradient { get; protected set; }

        protected Layer(
            InputOutputShape inputShape,
            int nodeCount,
            int randomSeed = 55)
        {
            NumNodes = nodeCount;
            RandomSeed = randomSeed;
            InputShape = inputShape;
        }

        public virtual void ResetAccumulators() { }

        public abstract Tensor FeedFoward(Tensor input);
        public abstract Tensor BackPropagation(Tensor dE_dY);
        public virtual void UpdateWeightsAndBiasesWithScaledGradients(float learningRate) { }
    }
}
