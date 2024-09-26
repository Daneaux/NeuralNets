using MatrixLibrary;
using System.Diagnostics;

namespace NeuralNets.Network
{
    // Note: here we use "weights" but that's really filters, the little NxN matrices we use to convole the incoming stacked matrices
    // note bug bug: Inputs can be lists of matrices of depth N, which means a list of a stack of matrices.
    public class ConvolutionLayer : Layer
    {
        public int FilterSize { get; }

        public List<AvxMatrix> Biases { get; }
        public List<AvxMatrix> Weights { get; private set; }

        public ConvolutionLayer(
            int nodeCount,
            int filterSize,
            IActivationFunction activationFunction,
            int incomingDataPoints,
            int randomSeed = 12341324) :
            base(nodeCount, activationFunction, incomingDataPoints, randomSeed)
        {
            FilterSize = filterSize;
            for (int i = 0; i < nodeCount; i++)
            {
                AvxMatrix filter = new AvxMatrix(filterSize, filterSize);
                filter.SetRandom(randomSeed, (float)-Math.Sqrt(nodeCount), (float)Math.Sqrt(nodeCount)); // Xavier initilization
                Weights.Add(filter);

                AvxMatrix bias = new AvxMatrix(filterSize, filterSize); // not sure these are the right dimensions.  TODO
                bias.SetRandom(randomSeed, -1f, 1f);
                Biases.Add(bias);
            }
        }

        public List<AvxMatrix> Activate(List<AvxMatrix> inputs)
        {
            List<AvxMatrix> activation = new List<AvxMatrix>();
            foreach (AvxMatrix outputMat in inputs)
            {
                var act = ActivationFunction.Activate(outputMat);
                activation.Add(act);
            }
            return activation;
        }

        public List<AvxMatrix> Derivative(List<AvxMatrix> lastActivation)
        {
            throw new NotImplementedException();
        }
        public override Tensor FeedFoward(Tensor input)
        {
            ConvolutionTensor convTensor = input as ConvolutionTensor;
            if (convTensor == null)
            {
                throw new ArgumentException("Expected a ConvolutionTensor");
            }

            var ret = FeedForwardConvolutionPlusBiasPlusActivation(convTensor.Matrices);
            return new ConvolutionTensor(ret);
        }

        // The input to the this layer is always a stack. In a CNN we don't receive different inputs
        // like a traditional ANN (where all the input edges are individually weighted).
        // Each node in the convolution layer receives the same input.
        // the 'weights' in this layer are effectively the filters (cnovolution filter) that are different per node
        private List<AvxMatrix> FeedForwardConvolutionPlusBiasPlusActivation(List<AvxMatrix> stack)
        {
            Debug.Assert(stack != null);

            List<AvxMatrix> output = new List<AvxMatrix>();
            for (int i = 0; i < NumNodes; i++)
            {
                AvxMatrix filter = Weights[i];
                AvxMatrix bias = Biases[i];
                AvxMatrix conv = ConvolveStack(stack, filter);
                output.Add(conv + bias);
            }

            List<AvxMatrix> activation = Activate(output);
            return activation;
        }

        private static AvxMatrix ConvolveStack(List<AvxMatrix> stack, AvxMatrix filter)
        {
            (int r, int c) = AvxMatrix.ConvolutionSizeHelper(stack[0], filter);
            AvxMatrix result = new AvxMatrix(r, c);
            foreach (AvxMatrix lhs in stack)
            {
                result += lhs.Convolution(filter);
            }
            return result;
        }

        public override void UpdateWeightsAndBiasesWithScaledGradients(Tensor weightGradient, Tensor biasGradient)
        {
            ConvolutionTensor ctBiases = biasGradient as ConvolutionTensor;
            ConvolutionTensor ctWeights = weightGradient as ConvolutionTensor;
            if (ctWeights == null || ctBiases == null)
            {
                throw new ArgumentException("Expectd ConvolutionTensor");
            }

            // todo: figure out how to updates "weights" and biases for convoultions.
        }
    }
}
