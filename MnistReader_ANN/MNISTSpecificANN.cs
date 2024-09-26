using NeuralNets;

namespace MnistReader_ANN
{
    public class MNISTSpecificANN : GeneralFeedForwardANN
    {
        public MNISTSpecificANN(float trainingRate, int inputDim, int outputDim) : base(trainingRate, inputDim, outputDim) //, 128, new MNISTTrainingSet())
        {
            this.LossFunction = new CategoricalCrossEntropy();
            Layers.Add(new WeightedLayer(128, new ReLUActivaction(), InputDim));
            Layers.Add(new WeightedLayer(64, new SigmoidActivation(), 128));
            Layers.Add(new WeightedLayer(32, new SigmoidActivation(), 64));
            Layers.Add(new WeightedLayer(OutputDim, new SoftMax(), 32));
        }
    }
}
