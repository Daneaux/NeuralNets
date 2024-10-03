using MatrixLibrary;
using System.Diagnostics;

namespace NeuralNets
{
    public class ConvolutionNN : NeuralNetworkAbstract
    {
        protected ConvolutionNN(float trainingRate, int inputDim, int outputDim)
        {
            this.LearningRate = trainingRate;
            this.InputDim = inputDim;
            this.OutputDim = outputDim;
        }

        public ConvolutionNN(
            List<Layer> layers,
            float trainingRate,
            int inputDim,
            int outputDim,
            ILossFunction lossFunction) : this(trainingRate, inputDim, outputDim)
        {
            Debug.Assert(layers != null);
            Debug.Assert(layers.Count > 0);
            lossFunction ??= new SquaredLoss();
            this.LossFunction = lossFunction;
            this.Layers = layers;
        }

        public override float GetTotallLoss(TrainingPair tp, AvxColumnVector predicted)
        {
            AvxColumnVector lossVec = this.LossFunction.Error(tp.Output, predicted);
            return lossVec.Sum();
        }

        public override AvxColumnVector GetLossVector(TrainingPair tp, AvxColumnVector predicted)
        {
            return this.LossFunction.Error(tp.Output, predicted);
        }

        public override float GetAveragelLoss(TrainingPair tp, AvxColumnVector predicted)
        {
            return this.GetTotallLoss(tp, predicted) / (float)predicted.Size;
        }
    }
}
