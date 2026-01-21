using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System.Diagnostics;

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
        public ColumnVectorBase? X { get; private set; }
        public ColumnVectorBase Y { get; private set; }
        public MatrixBase Weights { get; set; }
        public ColumnVectorBase Biases { get; set; }

        private List<MatrixBase> accumulatedWeights = new List<MatrixBase>();
        private List<ColumnVectorBase> accumulatedBiases = new List<ColumnVectorBase>();

        public override InputOutputShape OutputShape => new InputOutputShape(1, NumNodes, 1, 1);

        public WeightedLayer(
            InputOutputShape inputShape,
            int nodeCount, 
            int randomSeed = 55) : base(inputShape, nodeCount, randomSeed)
        {
            Biases = MatrixFactory.CreateColumnVector(nodeCount);
            Weights = MatrixFactory.CreateMatrix(nodeCount, inputShape.TotalFlattenedSize);
            
            this.Weights.SetRandom(randomSeed, (float)-Math.Sqrt(nodeCount), (float)Math.Sqrt(nodeCount)); // Xavier initilization
            this.Biases.SetRandom(randomSeed, -1, 10);
            
            Debug.Assert(this.Weights.Rows == this.Biases.Size);
            Debug.Assert(this.Weights.Cols == this.InputShape.TotalFlattenedSize);
        }

        public WeightedLayer(
            InputOutputShape inputShape,
            int nodeCount,
            MatrixBase initialWeights,
            ColumnVectorBase initialBiases) : base(inputShape, nodeCount)
        {
            this.Biases = initialBiases;
            this.Weights = initialWeights;
            Debug.Assert(this.Weights.Rows == this.Biases.Size);
            Debug.Assert(this.Weights.Cols == this.InputShape.TotalFlattenedSize);
        }

        public override Tensor FeedFoward(Tensor input)
        {
            AnnTensor annTensor = input as AnnTensor;
            ColumnVectorBase ?vectorInput = null;
            if (annTensor != null)
            {
                vectorInput = annTensor.ColumnVector;
                Debug.Assert(annTensor.Matrix == null);
            }
            else 
            {
                vectorInput = input.ToFlattenedMatrices();
            }
            this.X = vectorInput;
            this.Y = (Weights * X) + Biases;

            return new AnnTensor(null, Y);
        }

        public override Tensor BackPropagation(Tensor dE_dY)
        {
            // We want 3 gradients:
            // DE/DX (gradient of my Inputs)
            // DE/DW (gradient of my Weights)
            // DE/DB (gradient of my Baises)
            // Remember:  Y = W * X + B
            // Y : output
            // X : input
            // W : weights
            // B : Biases

            // Y = WX + B
            // We have DE/DY (this is the input to the method)
            // We want DE/DW
            // DE/DW = DE/DY * DY/DW
            //       = DE/DY * X
            MatrixBase weightGradient = X.RhsOuterProduct(dE_dY); // todo: ugly hack
            //MatrixBase weightGradient = dE_dY.ToAvxColumnVector().OuterProduct(X);
            ColumnVectorBase biasGradient = dE_dY.ToColumnVector();
            this.AccumulateGradients(weightGradient, biasGradient);

            // Now build De/Dx
            // De/Dx = De/Dy * Dy/Dx = De/Dy * W
            ColumnVectorBase dE_dX = this.Weights.GetTransposedMatrix() * dE_dY.ToColumnVector();
            return new AnnTensor(null, dE_dX);
        }

        public override void UpdateWeightsAndBiasesWithScaledGradients(float learningRate)
        {
            // add up and average all the gradients
            MatrixBase averageWeights = accumulatedWeights[0];
            for(int i = 1; i < accumulatedWeights.Count; i++)
            {
                averageWeights += accumulatedWeights[i];
            }
            averageWeights = averageWeights * ( learningRate / (float)accumulatedWeights.Count);

            ColumnVectorBase avgBiases = accumulatedBiases[0];
            for(int i = 1;i < accumulatedBiases.Count; i++)
            {
                avgBiases += accumulatedBiases[i];
            }
            avgBiases = avgBiases * (learningRate / (float)accumulatedBiases.Count);

            Biases -= avgBiases;
            Weights -= averageWeights;
        }

        public override void ResetAccumulators()
        {
            accumulatedBiases.Clear();
            accumulatedWeights.Clear();
        }
        private void AccumulateGradients(MatrixBase weightGradient, ColumnVectorBase biasGradient)
        {
            accumulatedBiases.Add(biasGradient);
            accumulatedWeights.Add(weightGradient);
        }
    }
}
