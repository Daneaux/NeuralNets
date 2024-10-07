using NeuralNets;

namespace MnistReader_ANN
{
    public class MNISTSpecificANN : GeneralFeedForwardANN
    {
        public MNISTSpecificANN(float trainingRate, int inputDim, int outputDim) : base(trainingRate, inputDim, outputDim) //, 128, new MNISTTrainingSet())
        {
            //this.LossFunction = new CategoricalCrossEntropy();
            this.LossFunction = new SquaredLoss();
            Layers.Add(new WeightedLayer(new MatrixLibrary.InputOutputShape(1, InputDim, 1, 1), 128, new ReLUActivaction()));
            Layers.Add(new WeightedLayer(new MatrixLibrary.InputOutputShape(1, 128, 1, 1), 64, new SigmoidActivation()));
            Layers.Add(new WeightedLayer(new MatrixLibrary.InputOutputShape(1, 64, 1, 1), 32, new SigmoidActivation()));
            Layers.Add(new WeightedLayer(new MatrixLibrary.InputOutputShape(1, 32, 1, 1), OutputDim, new SigmoidActivation()));
//            Layers.Add(new WeightedLayer(OutputDim, new SoftMax(), 32));
        }
    }
}
