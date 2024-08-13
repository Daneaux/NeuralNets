using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    /*
     * 
     * https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
     *
     */
    class Solver
    {
        public Network Net { get; protected set; }
    }

    // output: numerical
    // Output Activation Function: Linear
    // Loss Function: Mean Squared Error
/*    class RegressionPredictSingle : Solver 
    { 
        // for now: input, inner, output
        public RegressionPredictSingle(int inputCount, int hiddenLayerCount)
        {
            List<Layer> layers = new List<Layer>();
            layers.Add(new InputLayer(inputCount));
            layers.Add(new WeightedLayer(hiddenLayerCount));
            layers.Add(new RegressionOutputLayer(1));
            Net = new Network(layers, new SquaredLoss());
        }
    }*/

    class BinaryClassification : Solver { }

    class SingleLableMultiClass : Solver { }

    class MultiLabelMultiClass : Solver { }


}
