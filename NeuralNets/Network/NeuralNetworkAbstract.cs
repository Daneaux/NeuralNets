using MatrixLibrary;

namespace NeuralNets
{
    public abstract class NeuralNetworkAbstract
    {
        public int LayerCount => Layers.Count;
        public float LearningRate { get; protected set; }
        public ILossFunction LossFunction { get; protected set; }
        public List<Layer> Layers { get; protected set;  } = new List<Layer>();

        public Layer OutputLayer => Layers[LayerCount - 1];
        public int BatchSize { get; protected set; }
        public int InputDim { get; protected set; }
        public int OutputDim { get; protected set; }

        public abstract float GetAveragelLoss(TrainingPair tp, AvxColumnVector predicted);
        public abstract AvxColumnVector GetLossVector(TrainingPair tp, AvxColumnVector predicted);
        public abstract float GetTotallLoss(TrainingPair tp, AvxColumnVector predicted);
    }
}