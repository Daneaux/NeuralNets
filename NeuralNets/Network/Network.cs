using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public class Network
    {
        public int Seed { get; }
        public List<Layer> Layers { get; }
        public int LayerCount { get { return Layers.Count; } }
        public ILossFunction LossFunction { get; }
        public List<ColumnVector> Biases { get; private set; }
        public List<Matrix> WeightMatrices { get; private set; }

        public Network(List<Layer> layers, ILossFunction lossFunction, int randomSeed = 1234)
        {
            Seed = randomSeed;
            Layers = layers;
            LossFunction = lossFunction;
        }


        // todo: Save matricies so we can adjust during backprop
/*        public static ColumnVector FeedForward(InputLayer inputLayer, WeightedLayer[] layers)
        {
            ColumnVector activationVector = inputLayer.ValueVector;
            for(int i = 0; i < layers.Length; i++)
            {
                Matrix weightMatrix = layers[i].Weights;
                ColumnVector weightedSum =  (weightMatrix * activationVector) - layers[i].Biases;
                activationVector = DoActivation(weightedSum, layers[i]);
            }

            return activationVector;
        }*/

/*        public void BackProp(TrainingPair trainingPair, WeightedLayer[] layers)
        {
            ColumnVector outputVector = Network.FeedForward(new InputLayer(trainingPair.Input), layers);
            ColumnVector lossVector = this.LossFunction.Error(outputVector, trainingPair.Output);
            ColumnVector derivativeLossVector = this.LossFunction.Derivative(outputVector, trainingPair.Output);
        }*/

/*        private static ColumnVector DoActivation(ColumnVector inputColumnVector, WeightedLayer layer)
        {
            Debug.Assert(inputColumnVector.Size == layer.NumNodes);

            double[] activationOutVector = new double[inputColumnVector.Size];
            for(int i = 0; i < inputColumnVector.Size; i++)
            {
                activationOutVector[i] = layer.ActivationFunction.Activate(inputColumnVector[i]);
            }
            return new ColumnVector(activationOutVector);
        }*/
    }
}
