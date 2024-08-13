using NeuralNets;
using System.Diagnostics;

namespace NumReaderNetwork
{
    public class NumberReaderANN
    {
        public int InputDim { get; private set; }
        public int HiddenLayerDim { get; private set; }
        public int OutputLayerDim { get; private set; }
        //public InputLayer InputLayer { get; private set; }
        public List<WeightedLayer> WeightedLayers { get; private set; }

        public NumberReaderANN(int inputDim)//, int hiddenLayerDim, int outputLayerDim)
        { 
            this.InputDim = inputDim;
            this.OutputLayerDim = 10;
            this.HiddenLayerDim = 16;
        }

        public void InitializeNetwork()
        {
           // InputLayer = new InputLayer(InputDim);
            WeightedLayers.Add(new WeightedLayer(HiddenLayerDim, new ReLUActivaction(), InputDim));
            WeightedLayers.Add(new WeightedLayer(OutputLayerDim, new SigmoidActivation(), HiddenLayerDim));
        }

        public ColumnVector FeedForward(double[] inputs)
        {
            Debug.Assert(inputs.Length == this.InputDim);
            Debug.Assert(WeightedLayers.Count == 2);
            ColumnVector currentActivation = new ColumnVector(inputs);
            //InputLayer = new InputLayer(currentActivation); // doesn't see necessary at all?
            for (int i = 0; i < 2; i++)
            {
                WeightedLayer currentLayer = WeightedLayers[i];
                ColumnVector z1 = currentLayer.Weights * currentActivation;
                ColumnVector z12 = z1 - currentLayer.Biases;
                ColumnVector o1 = currentLayer.ActivationFunction.Activate(z12);
                currentActivation = o1;
            }
            return currentActivation;
        }

        public void BackProp()
        {
            // first find W0 - Wn in the hidden layer, just before the output layer
            // We want the derivative of the Error function in terms of the weights (w0 ... wn)
            // d(E)/dw = d(E)/o2 * d(o2)/z2 * d(z2)/w = (a-b) * 
            // d(E)/db = d(E)/o2 * d(o2)/z2 * d(z2)/b
            // 
        }


    }
}
