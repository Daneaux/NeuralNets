using NeuralNets.Network;
using NumReaderNetwork;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    /// <summary>
    /// This is purely a container of the current state of the network. It doesn't do any computation, it simply contains the network:
    /// Weights, Biases, Layers, LossFunction.  That's it.
    /// It's only operation is to update Weights and Biases
    /// 
    /// Not sure if it should contain TrainingRate (or learning rate) however. Keep it here for now since it's reponsible for updating and scaling weights and biases
    /// </summary>
    public class GeneralFeedForwardANN
    {
        public double TrainingRate { get; protected set; }
        public int BatchSize { get; protected set; }
        public int InputDim { get; protected set; }
        public int OutputDim { get; protected set; }
        public List<WeightedLayer> WeightedLayers { get; set; } = [];
        public int LayerCount { get {  return WeightedLayers.Count; } }
        public ILossFunction LossFunction { get; protected set; }
        public WeightedLayer OutputLayer { get { return WeightedLayers[WeightedLayers.Count - 1]; } }

        protected GeneralFeedForwardANN(double trainingRate, int inputDim, int outputDim)
        {
            this.TrainingRate = trainingRate;
            this.InputDim = inputDim;
            this.OutputDim = outputDim;
        }

        public GeneralFeedForwardANN(
            List<WeightedLayer> layers, 
            double trainingRate, 
            int inputDim,
            int outputDim,
            ILossFunction lossFunction) : this(trainingRate, inputDim, outputDim)
        {
            Debug.Assert(layers != null);
            Debug.Assert(layers.Count > 0);
            lossFunction ??= new SquaredLoss();
            this.LossFunction = lossFunction;
            this.WeightedLayers = layers;
        }

        public double GetTotallLoss(TrainingPair tp, ColumnVector predicted)
        {
            ColumnVector lossVec = this.LossFunction.Error(tp.Output, predicted);
            return lossVec.Sum();
        }

        public ColumnVector GetLossVector(TrainingPair tp, ColumnVector predicted)
        {
            return this.LossFunction.Error(tp.Output, predicted);
        }

        public double GetAveragelLoss(TrainingPair tp, ColumnVector predicted)
        {
            return this.GetTotallLoss(tp, predicted) / (double)predicted.Size;
        }
    }
}
