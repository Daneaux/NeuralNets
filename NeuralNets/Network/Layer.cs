using MatrixLibrary;
using NeuralNets.Network;
using System.Collections.Generic;
using System.Diagnostics;

namespace NeuralNets
{
    public abstract class Layer
    {
        public int IncomingDataPoints { get; }
        public int NumNodes { get; private set; }
        public IActivationFunction ActivationFunction { get; private set; }

        public abstract Tensor FeedFoward(Tensor input);

        protected Layer(int nodeCount, IActivationFunction activationFunction, int incomingDataPoints, int randomSeed = 12341324)
        {
            this.NumNodes = nodeCount;
            ActivationFunction = activationFunction;
            IncomingDataPoints = incomingDataPoints;
        }

        public abstract void UpdateWeightsAndBiasesWithScaledGradients(Tensor weightGradient, Tensor biasGradient);
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
        public AvxMatrix Weights { get; set; }
        public AvxColumnVector Biases { get; set; }

        public bool IsSoftMaxActivation
        {
            get
            {
                return this.ActivationFunction is SoftMax;
            }
        }

        public WeightedLayer(int nodeCount, IActivationFunction activationFunction, int incomingDataPoints, int randomSeed = 12341324) : base(nodeCount, activationFunction, incomingDataPoints, randomSeed)
        {
            this.Initialize(activationFunction, incomingDataPoints, new AvxMatrix(nodeCount, incomingDataPoints), new AvxColumnVector(nodeCount));
            this.Weights.SetRandom(randomSeed, (float)-Math.Sqrt(nodeCount), (float)Math.Sqrt(nodeCount)); // Xavier initilization
            this.Biases.SetRandom(randomSeed, -1, 10);
        }

        public WeightedLayer(
            int nodeCount,
            IActivationFunction activationFunction,
            int incomingDataPoints,
            AvxMatrix initialWeights,
            AvxColumnVector initialBiases) : base(nodeCount, activationFunction, incomingDataPoints)
        {
            Initialize(activationFunction, incomingDataPoints, initialWeights, initialBiases);
        }

        private void Initialize(IActivationFunction activationFunction, int incomingDataPoints, AvxMatrix initialWeights, AvxColumnVector initialBiases)
        {
            this.Weights = initialWeights;
            this.Biases = initialBiases;

            Debug.Assert(activationFunction != null);
            Debug.Assert(this.Weights.Rows == this.Biases.Size);
            Debug.Assert(this.Weights.Cols == incomingDataPoints);
        }

        public AvxColumnVector Activate(AvxColumnVector input)
        {
            return ActivationFunction.Activate(input);
        }

        public AvxColumnVector Derivative(AvxColumnVector lastActivation)
        {
            AvxColumnVector derivative = ActivationFunction.Derivative(lastActivation);
            return derivative;
        }

        public override Tensor FeedFoward(Tensor input)
        {
            AnnTensor annTensor = input as AnnTensor;
            if(annTensor == null)
            {
                throw new ArgumentException("expected AnnTensor as input");
            }
            AvxColumnVector vectorInput = annTensor.ColumnVector;
            AvxColumnVector Z = Weights * vectorInput + Biases;
            AvxColumnVector O = this.ActivationFunction.Activate(Z);
            return new AnnTensor(null, O);
        }

        public override void UpdateWeightsAndBiasesWithScaledGradients(Tensor weightGradient, Tensor biasGradient)
        {
            AnnTensor ctBiases = biasGradient as AnnTensor;
            AnnTensor ctWeights = weightGradient as AnnTensor;
            if (ctWeights == null || ctBiases == null)
            {
                throw new ArgumentException("Expectd ConvolutionTensor");
            }

            this.UpdateWeightsAndBiasesWithScaledGradients(ctWeights.Matrix, ctBiases.ColumnVector);
        }

        private void UpdateWeightsAndBiasesWithScaledGradients(AvxMatrix weightGradient, AvxColumnVector biasGradient)
        {
            Weights = Weights - weightGradient;
            Biases = Biases - biasGradient;
        }
    }
}
