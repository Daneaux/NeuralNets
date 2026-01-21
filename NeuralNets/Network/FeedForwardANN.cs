using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System.Diagnostics;

namespace NeuralNets
{
    /// <summary>
    /// This is purely a container of the current state of the network. It doesn't do any computation, it simply contains the network:
    /// Weights, Biases, Layers, LossFunction.  That's it.
    /// It's only operation is to update Weights and Biases
    /// 
    /// Not sure if it should contain TrainingRate (or learning rate) however. Keep it here for now since it's reponsible for updating and scaling weights and biases
    /// </summary>
    public class GeneralFeedForwardANN : NeuralNetworkAbstract
    {

        protected GeneralFeedForwardANN(float trainingRate, int inputDim, int outputDim)
        {
            this.LearningRate = trainingRate;
            this.InputDim = inputDim;
            this.OutputDim = outputDim;
        }

        public GeneralFeedForwardANN(
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

        public override float GetTotallLoss(TrainingPair tp, ColumnVectorBase predicted)
        {
            ColumnVectorBase lossVec = this.LossFunction.Error(tp.Output, predicted.ToTensor());
            return lossVec.Sum();
        }

        public override ColumnVectorBase GetLossVector(TrainingPair tp, ColumnVectorBase predicted)
        {
            return this.LossFunction.Error(tp.Output, predicted.ToTensor());
        }

        public override float GetAveragelLoss(TrainingPair tp, ColumnVectorBase predicted)
        {
            return this.GetTotallLoss(tp, predicted) / (float)predicted.Size;
        }
    }
}
