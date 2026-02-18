using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System.Diagnostics;

namespace NeuralNets
{
    /// <summary>
    /// This is purely a container of the current state of the network. It doesn't do any computation, it simply contains the network:
    /// Layers, LossFunction, some meta data. 
    /// 
    /// Why do we have an abstract class with one subclass? need to clean it up. there's no reason to have both.
    /// </summary>
    public class GeneralFeedForwardANN : NeuralNetworkAbstract
    {
        // deep copy
        public GeneralFeedForwardANN(GeneralFeedForwardANN srcNetwork)
        {
            this.InputDim = srcNetwork.InputDim;
            this.OutputDim = srcNetwork.OutputDim;
            this.LearningRate = srcNetwork.LearningRate;
            this.LossFunction = srcNetwork.LossFunction;
            this.Layers = DeepCopyLayers(srcNetwork.Layers);
        }

        private List<Layer> DeepCopyLayers(List<Layer> layers)
        {
            List<Layer> copiedLayers = new List<Layer>(layers.Count);
            for (int i=0; i < layers.Count; i++)
            {
                copiedLayers.Add(layers[i].DeepCopy());
            }
            return copiedLayers;
        }

        protected GeneralFeedForwardANN(float trainingRate, int inputDim, int outputDim)
        {
            this.InputDim = inputDim;
            this.OutputDim = outputDim;
            this.LearningRate = trainingRate;
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
            return this.LossFunction.Error(tp.Output, predicted.ToTensor());
        }


        public override float GetAveragelLoss(TrainingPair tp, ColumnVectorBase predicted)
        {
            return this.GetTotallLoss(tp, predicted) / (float)predicted.Size;
        }
    }
}
