using MatrixLibrary;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
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

        public override InputOutputShape OutputShape => new InputOutputShape(1, NumNodes, 1, 1);

        public WeightedLayer(
            InputOutputShape inputShape,
            int nodeCount, 
            IActivationFunction activationFunction, 
            int randomSeed = 12341324) : base(inputShape, nodeCount, activationFunction, randomSeed)
        {
            Biases = new AvxColumnVector(nodeCount);
            Weights = new AvxMatrix(nodeCount, inputShape.TotalFlattenedSize);
            
            this.Weights.SetRandom(randomSeed, (float)-Math.Sqrt(nodeCount), (float)Math.Sqrt(nodeCount)); // Xavier initilization
            this.Biases.SetRandom(randomSeed, -1, 10);

            Debug.Assert(activationFunction != null);
            Debug.Assert(this.Weights.Rows == this.Biases.Size);
            Debug.Assert(this.Weights.Cols == this.InputShape.TotalFlattenedSize);
        }

        public WeightedLayer(
            InputOutputShape inputShape,
            int nodeCount,
            IActivationFunction activationFunction,
            AvxMatrix initialWeights,
            AvxColumnVector initialBiases) : base(inputShape, nodeCount, activationFunction)
        {
            this.Biases = initialBiases;
            this.Weights = initialWeights;
            Debug.Assert(activationFunction != null);
            Debug.Assert(this.Weights.Rows == this.Biases.Size);
            Debug.Assert(this.Weights.Cols == this.InputShape.TotalFlattenedSize);
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
            AvxColumnVector ?vectorInput = null;
            if (annTensor != null)
            {
                vectorInput = annTensor.ColumnVector;
                Debug.Assert(annTensor.Matrix == null);
            }
            else 
            {
                vectorInput = input.ToFlattenedMatrices();
            }

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
