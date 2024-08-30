using MnistReader_ANN;
using NeuralNets;

namespace NumReaderNetwork
{
    public class MNISTSpecificANN : GeneralFeedForwardANN
    {
        public MNISTSpecificANN(double trainingRate) : base(trainingRate, 128, new MNISTTrainingSet())
        {
            this.LossFunction = new CategoricalCrossEntropy();
            WeightedLayers.Add(new WeightedLayer(128, new ReLUActivaction(), InputDim));
            WeightedLayers.Add(new WeightedLayer(64, new SigmoidActivation(), 128));
            WeightedLayers.Add(new WeightedLayer(32, new SigmoidActivation(), 64));
            WeightedLayers.Add(new WeightedLayer(OutputDim, new SoftMax(), 32));
        }
    }
}
