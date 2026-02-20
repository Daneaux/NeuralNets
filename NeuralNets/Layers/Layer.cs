using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    public abstract class Layer
    {
        public int NodeCount { get; private set; }
        public int RandomSeed { get; }
        public InputOutputShape InputShape { get; }
        public InputOutputShape OutputShape { get; protected set; }
        public virtual int AccumulationCount { get; protected set; }

        public MatrixBase LastWeightGradient { get; protected set; }
        public ColumnVectorBase LastBiasGradient { get; protected set; }

        protected Layer(
            InputOutputShape inputShape,
            int nodeCount,
            int randomSeed = 55)
        {
            NodeCount = nodeCount;
            RandomSeed = randomSeed;
            InputShape = inputShape;
        }

        // Deep copy, except for matrices those aren't necessary.
        protected Layer(Layer srcLayer)
        {
            this.NodeCount = srcLayer.NodeCount;
            this.RandomSeed = srcLayer.RandomSeed;
            this.InputShape = srcLayer.InputShape;
            this.OutputShape = srcLayer.OutputShape;
            this.LastWeightGradient = null;
            this.LastBiasGradient = null;
        }

        public abstract Layer DeepCopy();

        public abstract void Initialize();

        public virtual void ResetAccumulators() { }

        public abstract Tensor FeedFoward(Tensor input);
        public abstract Tensor BackPropagation(Tensor dE_dY);
        public virtual void UpdateWeightsAndBiasesWithScaledGradients(float learningRate) { }
        internal virtual void AccumulateGradientsFrom(Layer layer) { }
        internal virtual void CopyWeightsAndBiasesFrom(Layer layer) { }
    }
}
