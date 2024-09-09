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
        public bool IsSoftMaxActivation
        {
            get
            {
                return this.ActivationFunction is SoftMax;
            }
        }

        protected IActivationFunction ActivationFunction {  get; private set; }
        public WeightedLayer(int nodeCount, IActivationFunction activationFunction, int incomingDataPoints, int randomSeed=12341324) : base(nodeCount)
        {
            this.Initialize(activationFunction, incomingDataPoints, new Matrix2D(nodeCount, incomingDataPoints), new ColumnVector(nodeCount));
            this.Weights.SetRandom(randomSeed, (float)-Math.Sqrt(nodeCount), (float)Math.Sqrt(nodeCount)); // Xavier initilization
            this.Biases.SetRandom(randomSeed, -1, 10);
        }

        public WeightedLayer(
            int nodeCount, 
            IActivationFunction activationFunction, 
            int incomingDataPoints,
            Matrix2D initialWeights,
            ColumnVector initialBiases) : base(nodeCount)
        {
            Initialize(activationFunction, incomingDataPoints, initialWeights, initialBiases);
        }

        private void Initialize(IActivationFunction activationFunction, int incomingDataPoints, Matrix2D initialWeights, ColumnVector initialBiases)
        {
            this.Weights = initialWeights;
            this.Biases = initialBiases;

            Debug.Assert(activationFunction != null);
            this.ActivationFunction = activationFunction;

            Debug.Assert(this.Weights.Rows == this.Biases.Size);
            Debug.Assert(this.Weights.Cols == incomingDataPoints);

           // this.LastSigma = null;
           // this.BiasGradient = new ColumnVector(this.NumNodes);
           // this.WeightGradient = new Matrix(this.Weights.Rows, this.Weights.Cols);
        }

        public ColumnVector Activate(ColumnVector input)
        {
            return ActivationFunction.Activate(input);
        }

        public ColumnVector Derivative(ColumnVector lastActivation)
        {
            ColumnVector derivative = ActivationFunction.Derivative(lastActivation);

            return derivative;
        }

        // Weight matrix is for one sample, and the number of rows corresponds to the number of hidden layer nodes, for example 16.
        // And the number of columns is the number of data points in a samples, for examle 768 b&w pixel values for the MNIST number set
        public Matrix2D Weights { get; set; }
        public ColumnVector Biases { get; set; }


        public void UpdateWeightsAndBiasesWithScaledGradients(Matrix2D weightGradient, ColumnVector biasGradient)
        {
            Weights = Weights - weightGradient;
            Biases = Biases - biasGradient;
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
