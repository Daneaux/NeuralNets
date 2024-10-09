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
            (int r, int c) = AvxMatrix.ConvolutionSizeHelper(inputShape, FilterSize, isFull: false, stride);
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

        public List<AvxMatrix> Derivative(List<AvxMatrix> derivativeE_wrt_Y)
        {
            // Y = output = X ** K + B  (Input convolve with Kernel + Bias)
            // X = Input to the convolution layer. For example, an image
            // K = Kernel
            // B = Bias
            // Z = Activation(Y)

            /* Credit:  https://www.youtube.com/watch?v=Lakz2MoHy6o
             * Example output variable:
             * y11 = b11 + k11 * X11 + K12 * X12 + K21 * x21 + K22 * X22
             * y12 = b12 + k11 * X12 + K12 * X13 + K21 * x22 + K22 * X23
             * y21 = b21 + k11 * X21 + K12 * X22 + K21 * x31 + K22 * X32
             * y22 = b22 + k11 * X22 + K12 * X23 + K21 * x32 + K22 * X33
             * 

            - We want DE / DK
            - DE / DK = DE / DY * DY / DK

            So for the example above:
            * DE / Dk11 = DE/Dy11 * Dy11/Dk11 +   DE/Dy12 * Dy12/Dk11   +   DE/Dy21 * Dy21/Dk11 + DE/Dy22 * Dy22/Dk11
            *           = De/Dy11 * X11       +   De/Dy12 * X12         +   De/Dy21 * X21       + De/Dy22 * X22
            *           
            * (do all the rows)
            * And we realize that this is the cross correlation between X (input matrix) and DE/DY (Output gradient)
            * DE/DK11 = X1 <cross correlated> DE/DY1
            * 
            * Remember, however, that the full feed forward convolution equation is
            * 'd' inputs (for example an images with X depth, is x matrices, each the same size)
            * So we have X1 ... Xd inputs.
            * For the input we have N corresponding Kernels, each Kernel has the same depth (so if the input image is a 28x28 x3 (3 depth), then each kernel is 3 deep as well
            * And we have N Kernels.
            * For each Kernel stack, we have one Bias
            * Since we have N kernel stacks, we therefore have N outputs. Y1 ... YN
            * So we need the E/K for all K's (remember there's 3 K's per stack, and N Kernels.
            * If you think of all the kernels as a matrix (let's say each 'stack' is 3 deep, and there's N kernels)
            * ------------
            *  K11 K12 K12
            *  K21 K22 K23
            *  K31 K32 K33
            *  K41 ... ...
            *  ... ... ...
            *  KN1 KN2 KN3
            * ------------
            * 
            * From above
            * DE/Dk11 = X1 ** DE/Dy1
            * DE/Dk43 = X4 ** DE/Dy3
            * etc...
            * DE / D(Kij) = Xj ** DE/D(Yi)
            */


            /* -- BIAS Derivative
             * DE / DBi = DE / DYi
             */

            /* -- DE / DX
             * De/Dx11 = De/Dy11*K11
             * De/Dx12 = De/y11*K12 + De/Dy12 * K11
             * ...
             * 
             * De/Dx = the FULL Convolution: DE **Full** K (rotation 180degrees)
             * 
             * DE/DXj = Sigma(i=1 .. d)[De/DYi **full** Kij, j = 1...n
             */
            return null;
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
