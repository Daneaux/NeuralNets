using MatrixLibrary;
using System.Diagnostics;

namespace NeuralNets.Network
{
    public class ConvolutionRenderContext
    {
        public ConvolutionNN Network { get; }
        public int BatchSize { get; }
        public int CurrentThreadID { get; private set; }
        public AvxColumnVector[] Sigma { get; private set; }
        public Tensor[] ActivationContext { get; }
        public Tensor[] DerivativeContext { get; }
        public bool DoRandomSamples { get; private set; }
        public virtual ITrainingSet TrainingSet { get; }

        public int InputDim => Network.InputDim;
        public int OutputDim => Network.OutputDim;
        public int LayerCount => Network.LayerCount;
        public float LearningRate => Network.LearningRate;
        public Layer OutputLayer => Network.OutputLayer;
        public ILossFunction LossFunction => Network.LossFunction;
        public List<Layer> Layers => Network.Layers;

        public Tensor[] WeightGradient { get; }
        public Tensor[] BiasGradient { get; }

        public ConvolutionRenderContext (ConvolutionNN network, int batchSize, ITrainingSet trainingSet)
        {
            Network = network;
            this.BatchSize = batchSize;
            this.TrainingSet = trainingSet;
            //this.Sigma = new AvxColumnVector[this.LayerCount];
            //this.WeightGradient = new AvxMatrix[this.LayerCount];
            //this.BiasGradient = new AvxColumnVector[this.LayerCount];
            this.ActivationContext = new Tensor[this.LayerCount];
            this.DerivativeContext = new Tensor[this.LayerCount];
        }

        public static void BatchTrain(ConvolutionRenderContext parentContext, int epochNum)
        {

        }

        public Tensor FeedForward(Tensor input)
        {
            // if convolution layer then convole, add bias, do activation
            // then the output becomes a stack to the next layer.
            // if next layer is convolution, then everyone gets the same stack
            // if next layer is fully connected, then stack becomes a long array

            var lastActivation = input;
            for (int i = 0; i < this.LayerCount; i++)
            {
                Layer currentLayer = Layers[i];
                Tensor output = currentLayer.FeedFoward(lastActivation);
                lastActivation = output;

                this.SetLastActivation(i, lastActivation);
            }
            return lastActivation;
        }
        private void SetLastActivation(int layerIndex, Tensor lastActivation)
        {
            Debug.Assert(layerIndex >= 0);
            ActivationContext[layerIndex] = lastActivation;
        }
    }
}
