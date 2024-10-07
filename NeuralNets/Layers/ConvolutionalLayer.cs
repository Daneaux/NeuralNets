using MatrixLibrary;
using MatrixLibrary.Avx;
using System.Diagnostics;
using System.Numerics;

namespace NeuralNets
{
    // Note: here we use "weights" but that's really filters, the little NxN matrices we use to convole the incoming stacked matrices
    // note bug bug: Inputs can be lists of matrices of depth N, which means a list of a stack of matrices.
    public class ConvolutionLayer : Layer
    {
        private readonly int KernelDepth;
        private readonly int KernelCount;

        public int FilterSize { get; }
        public int Stride { get; }
        public List<AvxMatrix> Biases { get; } = new List<AvxMatrix>();
        public override InputOutputShape OutputShape { get; }
        public List<List<SquareKernel>> Kernels { get; private set; }

        public ConvolutionLayer(
            InputOutputShape inputShape,
            int kernelCount,
            int kernelSquareDimension,
            IActivationFunction activationFunction,
            int stride = 1,
            int randomSeed = 12341324) :
            base(inputShape, kernelCount, activationFunction, randomSeed)
        {
            Debug.Assert(inputShape.Count == 1);
            // I'm keeping my own copies of these properties even though they already exist on the base class
            // simply because my naming is specialized.
            KernelCount = kernelCount;
            KernelDepth = inputShape.Depth;
            FilterSize = kernelSquareDimension;
            Stride = stride;
            (int r, int c) = AvxMatrix.ConvolutionSizeHelper(inputShape, FilterSize, stride);
            OutputShape = new InputOutputShape(c, r, KernelDepth, kernelCount);
            Kernels = new List<List<SquareKernel>>(kernelCount);
            for (int i = 0; i < kernelCount; i++)
            {
                List<SquareKernel> kernelStack = new List<SquareKernel>(KernelDepth);
                for (int z = 0; z < KernelDepth; z++)
                {
                    SquareKernel filter = new SquareKernel(kernelSquareDimension);
                    filter.SetRandom(randomSeed, (float)-Math.Sqrt(kernelCount), (float)Math.Sqrt(kernelCount)); // Xavier initilization
                    kernelStack.Add(filter);
                }
                Kernels.Add(kernelStack);
                AvxMatrix bias = new AvxMatrix(kernelSquareDimension, kernelSquareDimension); // not sure these are the right dimensions.  TODO
                bias.SetRandom(randomSeed, -1f, 1f);
                Biases.Add(bias);
            }
        }
        public override Tensor FeedFoward(Tensor input)
        {
            List<AvxMatrix> avxMatrices = input.Matrices;
            Debug.Assert(avxMatrices != null);
            Debug.Assert(avxMatrices.Count == KernelDepth);

            var ret = FeedForwardConvolutionPlusBiasPlusActivation(avxMatrices);
            return new ConvolutionTensor(ret);
        }

        public List<AvxMatrix> Activate(List<AvxMatrix> input)
        {
            Debug.Assert(input.Count == KernelCount);
            List<AvxMatrix> activation = new List<AvxMatrix>();
            foreach (AvxMatrix outputMat in input)
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

        // The input to the this layer is always a stack. In a CNN we don't receive different inputs
        // like a traditional ANN (where all the input edges are individually weighted).
        // Each kernel in the convolution layer receives the same input.
        // Each kernel has N layers that matches the number of layers in the input.
        // For example if the input is an image with 3 layers (R, G, and B), then each Kernel has
        // 3 corresponding layers (3 square filters). And one bias.
        // Number of "kernels" refers to the total number of trainable kernels (each having 3 layers, ie: 3 filters)
        private List<AvxMatrix> FeedForwardConvolutionPlusBiasPlusActivation(List<AvxMatrix> inputStack)
        {
            Debug.Assert(inputStack != null);

            List<AvxMatrix> output = new List<AvxMatrix>();
            for (int i = 0; i < KernelCount; i++)
            {
                List<SquareKernel> filters = Kernels[i];
                AvxMatrix convolvedStack = ConvolveStacks(inputStack, filters);
                AvxMatrix Oh = convolvedStack + Biases[i];
                output.Add(Oh);
            }

            List<AvxMatrix> activation = Activate(output);
            Debug.Assert(activation.Count == KernelCount);
            return activation;
        }

        private static AvxMatrix ConvolveStacks(List<AvxMatrix> stack, List<SquareKernel> filters)
        {
            Debug.Assert(stack.Count == filters.Count);
            Debug.Assert(filters[0].Rows == filters[0].Cols);  // sanity check
            (int r, int c) = AvxMatrix.ConvolutionSizeHelper(stack[0], filters[0].FilterSize);
            AvxMatrix result = new AvxMatrix(r, c);
            for (int i = 0; i < stack.Count; i++)
            {
                result += stack[i].Convolution(filters[i]);
            }
            return result;
        }
    }
}
