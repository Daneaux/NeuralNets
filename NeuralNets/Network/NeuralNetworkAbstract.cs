using MatrixLibrary;

namespace NeuralNets
{
    public abstract class NeuralNetworkAbstract
    {
        public abstract int LayerCount { get; }
        public float LearningRate { get; protected set; }
        public ILossFunction LossFunction { get; protected set; }
        public abstract List<Layer> Layers { get; protected set;  }
        public abstract int BatchSize { get; protected set; }
        public abstract int InputDim { get; protected set; }
        public abstract int OutputDim { get; protected set; }
        public abstract float GetAveragelLoss(TrainingPair tp, AvxColumnVector predicted);
        public abstract AvxColumnVector GetLossVector(TrainingPair tp, AvxColumnVector predicted);
        public abstract float GetTotallLoss(TrainingPair tp, AvxColumnVector predicted);
    }
}