using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public class Layer
    {
        public int NumNodes { get; private set; }

        protected Layer(int nodeCount)
        {
            this.NumNodes = nodeCount;
        }
    }
    
    /// <summary>
    /// Weighter Layer contains:
    /// - The weights of all the incoming edges (weight matrix)
    /// - Biases
    /// - Last Activation ==> the last output of this layer, last time we ran feed forward
    /// - ScaledWeightDelta ==> during back propagation, we store the scaled (by learning rate) the weight delta matrix (aka: the results of all the partial derivatives to figure out the slope of each of the 'w' in terms of error
    /// - ScaledBiasDelta ==> Ditto for weights, except these are for biases.
    /// - Last Sigma ==> This is a reused computation during backprop. It's the product of the two column vectors: D(O)/D(Z) * D(Error)/D(O).  
    ///                  Which means the derivative of the error (or loss) in terms of the output of the node times the derivative of the activation function in terms of the input to the node
    /// </summary>
    public class WeightedLayer : Layer
    {
        public IActivationFunction ActivationFunction { get; protected set; }
        public WeightedLayer(int nodeCount, IActivationFunction activationFunction, int incomingDataPoints, int randomSeed=12341324) : base(nodeCount)
        {
            this.ActivationFunction = activationFunction;
            this.Weights = new Matrix(nodeCount, incomingDataPoints);
            this.Weights.SetRandom(randomSeed, - Math.Sqrt(nodeCount), Math.Sqrt(nodeCount)); // Xavier initilization
            this.Biases = new ColumnVector(nodeCount);
            this.Biases.SetRandom(randomSeed, -1.0, 1.0);      // todo: what should this range be?        
        }

        public WeightedLayer(
            int nodeCount, 
            IActivationFunction activationFunction, 
            int incomingDataPoints,
            Matrix initialWeights,
            ColumnVector initialBiases) : base(nodeCount)
        {
            Debug.Assert(activationFunction != null);
            this.Weights = initialWeights;
            this.Biases = initialBiases;
            Debug.Assert(this.Weights.Cols == this.Biases.Size);
            Debug.Assert(this.Weights.Rows == incomingDataPoints);
            this.ActivationFunction = activationFunction;
            this.ScaledWeightDelta = null;
            this.ScaledBiasDelta = null;
            this.LastSigma = null;
        }

        // Weight matrix is for one sample, and the number of rows corresponds to the number of hidden layer nodes, for example 16.
        // And the number of columns is the number of data points in a samples, for examle 768 b&w pixel values for the MNIST number set
        public Matrix Weights { get; set; }
        public ColumnVector Biases { get; set; }
        public ColumnVector? LastActivationOutput { get { return this.ActivationFunction.LastActivation; } }
        public Matrix ScaledWeightDelta { get; set; }
        public ColumnVector ScaledBiasDelta { get; set; }
        public ColumnVector LastSigma { get; set; }

        public void UpdateWeights()
        {
            Weights = Weights - this.ScaledWeightDelta;
        }

        public void UpdateBiases() 
        {
            Biases = Biases - this.ScaledBiasDelta;
        } 
        
        public ColumnVector GetActivationFunctionDerivative()
        {
            return this.ActivationFunction.Derivative();
        }

        public void StashDeltas(Matrix scaledGradientWeights, ColumnVector biasDelta)
        {
            this.ScaledWeightDelta = scaledGradientWeights;
            this.ScaledBiasDelta = biasDelta;
        }
    }

    public class InputLayer : Layer 
    { 
        public InputLayer(int nodeCount) : base(nodeCount)
        {
        }

        public InputLayer(ColumnVector valueVector) : base(valueVector.Size)
        {
            ValueVector = valueVector;
        }

        public ColumnVector ValueVector { get; internal set; }
    }

}
