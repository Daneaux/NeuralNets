using MatrixLibrary;
using System.Diagnostics;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    public class ConvolutionLayer : Layer
    {
        private readonly int KernelDepth;
        private readonly int KernelCount;

        public int Stride { get; }
        public int KernelSize { get; }
        //public List<AvxMatrix> Biases { get; private set; }
        public List<MatrixBase> Biases { get; private set; }
        public List<MatrixBase> BiasesGradientAccumulator { get; private set; }
        public override InputOutputShape OutputShape { get; }
        public KernelStacks Kernels { get; private set; }
        public KernelStacks KernelGradientAccumulator { get; private set; }
        public int AccumulationCount { get; private set; }
        private List<MatrixBase> LastInput { get; set; }

        public ConvolutionLayer(
            InputOutputShape inputShape,
            int kernelCount,
            int kernelSquareDimension,
            int stride = 1,
            int randomSeed = 55) :
            base(inputShape, kernelCount, randomSeed)
        {
            Debug.Assert(inputShape.Count == 1);
            // I'm keeping my own copies of these properties even though they already exist on the base class
            // simply because my naming is specialized.
            KernelCount = kernelCount;
            KernelDepth = inputShape.Depth;
            KernelSize = kernelSquareDimension;
            Stride = stride;

            (int r, int c) = MatrixLibrary.MatrixHelpers.ConvolutionSizeHelper(inputShape, KernelSize, isFull: false, stride);
            OutputShape = new InputOutputShape(c, r, KernelDepth, kernelCount);
            InitKernelsAndBiases();
            ResetAccumulators();
        }

        private void InitKernelsAndBiases()
        {
            Biases = new List<MatrixBase>(KernelCount);
            Kernels = new KernelStacks(KernelCount, KernelDepth, KernelSize);
            for (int i = 0; i < KernelCount; i++)
            {
                for (int j = 0; j < KernelDepth; j++)
                {
                    MatrixBase kernel = MatrixFactory.CreateMatrix(KernelSize, KernelSize);
                    kernel.SetRandom(RandomSeed, (float)-Math.Sqrt(KernelCount), (float)Math.Sqrt(KernelCount)); // Xavier initilization
                    Kernels[i, j] = kernel;
                }
                MatrixBase bias = MatrixFactory.CreateMatrix(OutputShape.Height, OutputShape.Width);
                bias.SetRandom(RandomSeed, -1f, 1f);
                Biases.Add(bias);
            }
        }

        public override Tensor FeedFoward(Tensor input)
        {
            List<MatrixBase> avxMatrices = input.Matrices;
            Debug.Assert(avxMatrices != null);
            Debug.Assert(avxMatrices.Count == KernelDepth);

            LastInput = avxMatrices;

            var ret = FeedForwardConvolutionPlusBias(avxMatrices);
            return new ConvolutionTensor(ret);
        }

        public override Tensor BackPropagation(Tensor dE_dX)
        {
            AccumulationCount++;
            (KernelStacks kernelGradients, 
                List<MatrixBase>? biasGradients, 
                List<MatrixBase>? inputGradients) = 
                this.Derivative(dE_dX.Matrices);
            AccumulateWeightsAndBiases(kernelGradients, biasGradients);
            return inputGradients.ToTensor();
        }

        public override void ResetAccumulators()
        {
            AccumulationCount = 0;
            KernelGradientAccumulator = new KernelStacks(KernelCount, KernelDepth, KernelSize);
            KernelGradientAccumulator.Reset();

            BiasesGradientAccumulator = new List<MatrixBase>();

            int r = Biases[0].Rows;
            int c = Biases[0].Cols;
            for (int i = 0; i < Biases.Count; i++)
                BiasesGradientAccumulator.Add(new AvxMatrix(r, c));
        }

        public override void UpdateWeightsAndBiasesWithScaledGradients(float learningRate)
        {
            KernelGradientAccumulator.ScaleAndAverage(AccumulationCount, learningRate);
            Kernels.Subtract(KernelGradientAccumulator);

            float scaleFactor = learningRate / (float)AccumulationCount;
            for (int i = 0; i < BiasesGradientAccumulator.Count; i++)
                BiasesGradientAccumulator[i] *= scaleFactor;

            for (int i = 0; i < Biases.Count; i++)
                Biases[i] -= BiasesGradientAccumulator[i];
        }

        private void AccumulateWeightsAndBiases(KernelStacks kernelGradients, List<MatrixBase> biasGradients)
        {
            lock (GradientLock)
            {
                KernelGradientAccumulator.Accumulate(kernelGradients);

                Debug.Assert(biasGradients.Count == Biases.Count);
                for (int i = 0; i < this.Biases.Count; i++)
                {
                    BiasesGradientAccumulator[i] += biasGradients[i];
                }
            }
        }

        public (KernelStacks kernelGradients, List<MatrixBase> biasGradients, List<MatrixBase> inputGradients) Derivative(List<MatrixBase> derivativeE_wrt_Y)
        {
            // Y = output = X ** K + B  (Input convolve with Kernel + Bias)
            // X = Input to the convolution layer. For example, an image
            // K = Kernel
            // B = Bias
            // Z = Activation(Y)
            // N = depth of input
            // D = number of kernel stacks in the layer, also called layer depth. D is the number of outputs Y (Y1 ... YD)

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
            * 'n' inputs (for example an images with N depth, is N matrices, each the same size)
            * So we have X1 ... XN inputs.
            * For the input we have N corresponding Kernels (one for each 'depth' of the input). So if the input image is a 28x28 x3 (3 depth), then each kernel is 3 deep as well
            * We also have D Kernels (each N deep). We can call each 'kernel' a kernel stack with N kernels in it.
            * For each Kernel stack, we have one Bias
            * Since we have D kernel stacks, we therefore have D outputs. Y1 ... YD
            * So we need the E/K for all K's (remember there's 3 Kernels's per stack), and D Kernels.
            * If you think of all the kernels as a matrix (let's say each 'stack' is 3 deep, and there's D kernels)
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
            * Here 'i' is the index of the kernel stack (1 ... D) and index of the output derivative we're given (Yi)
            * And 'j' is the index of the kernel inside each kernel stack (1 ... N)
            * 
            * So: Kernels.Count == D == DE/DY.Count = # of derivative matrices of Y from next layer
            * And: this.KernelDepth == N == Input count (number of matrices coming in as X)
            */
            Debug.Assert(this.Kernels.KernelCount == derivativeE_wrt_Y.Count);
            Debug.Assert(this.KernelDepth == this.LastInput.Count);

            List<List<MatrixBase>> kernelGradients = new List<List<MatrixBase>>();
            int N = this.KernelDepth;
            int D = this.KernelCount;
            for(int i = 0; i < D; i++)
            {
                kernelGradients.Add(new List<MatrixBase>());
                var inputs = LastInput;
                for(int j = 0; j < N; j++)
                {
                    MatrixBase dE_dY = derivativeE_wrt_Y[j];
                    Debug.Assert(dE_dY.IsSquare());
                    var gradientE_K = LastInput[j].Convolution(dE_dY);
                    kernelGradients[i].Add(gradientE_K);
                }
            }

            /* -- BIAS Derivative
             * DE / DBi = DE / DYi
             */
            List<MatrixBase> biasGradients = new List<MatrixBase>();
            foreach(MatrixBase dedy in derivativeE_wrt_Y)
            {
                biasGradients.Add(dedy);
            }


            /* -- DE / DX
             * De/Dx11 = De/Dy11*K11
             * De/Dx12 = De/y11*K12 + De/Dy12 * K11
             * ...
             * 
             * De/Dx = the FULL Convolution: DE **Full** K (rotation 180degrees)
             * 
             * DE/DXj = Sigma(i=1 .. d)[De/DYi **full** Kij, j = 1...n
             */

            List<MatrixBase> dEdXGradients = new List<MatrixBase>();
            (int r, int c) = MatrixLibrary.MatrixHelpers.ConvolutionSizeHelper(derivativeE_wrt_Y[0], this.Kernels.KernelSize, true);
            Debug.Assert(r == c);
            for (int k = 0; k < N; k++)
                dEdXGradients.Add(MatrixFactory.CreateMatrix(r,c));

            for (int i = 0; i < D; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    MatrixBase fullConvolve = derivativeE_wrt_Y[j].ConvolutionFull(this.Kernels[i, j]);
                    dEdXGradients[j] = dEdXGradients[j] + fullConvolve;
                }
            }

            return (new KernelStacks(kernelGradients), biasGradients, dEdXGradients);
        }

        // The input to the this layer is always a stack. In a CNN we don't receive different inputs
        // like a traditional ANN (where all the input edges are individually weighted).
        // Each kernel in the convolution layer receives the same input.
        // Each kernel has N layers that matches the number of layers in the input.
        // For example if the input is an image with 3 layers (R, G, and B), then each Kernel has
        // 3 corresponding layers (3 square filters). And one bias.
        // Number of "kernels" refers to the total number of trainable kernels (each having 3 layers, ie: 3 filters)
        private List<MatrixBase> FeedForwardConvolutionPlusBias(List<MatrixBase> inputStack)
        {
            Debug.Assert(inputStack != null);
            Debug.Assert(inputStack.Count == this.KernelDepth);

            List<MatrixBase> output = new List<MatrixBase>();
            for (int i = 0; i < KernelCount; i++)
            {
                MatrixBase convolvedStack = ConvolveStacks(inputStack, Kernels[i]);
                MatrixBase Oh = convolvedStack + Biases[i];
                output.Add(Oh);
            }

            return output;
        }

        private static MatrixBase ConvolveStacks(List<MatrixBase> inputStack, List<MatrixBase> kernelStack)
        {
            Debug.Assert(inputStack.Count == kernelStack.Count);
            Debug.Assert(kernelStack[0].Rows == kernelStack[0].Cols);  // sanity check
            Debug.Assert(kernelStack[0].IsSquare());
            (int r, int c) = MatrixLibrary.MatrixHelpers.ConvolutionSizeHelper(inputStack[0], kernelStack[0].Rows);
            MatrixBase result = MatrixFactory.CreateMatrix(r, c);
            for (int i = 0; i < inputStack.Count; i++)
            {
                Debug.Assert(kernelStack[i].IsSquare());
                result += inputStack[i].Convolution(kernelStack[i]);
            }
            return result;
        }
    }
}
