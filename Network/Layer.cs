using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    class Layer
    {
        public int NumNodes { get; private set; }
        public IActivationFunction ActivationFunction { get; protected set; }

        protected Layer(int nodeCount)
        {
            this.NumNodes = nodeCount;
        }
    }
    
    class InnerLayer : Layer
    {
        public InnerLayer(int nodeCount) : base(nodeCount)
        {
            this.ActivationFunction = new ReLUActivaction();
        }
    }

    class InputLayer : Layer 
    { 
        public InputLayer(int nodeCount) : base(nodeCount)
        {
            this.ActivationFunction = null;
        }
    }
    class RegressionOutputLayer : Layer 
    { 
        public RegressionOutputLayer(int nodeCount) : base(nodeCount)
        {
            this.ActivationFunction = new ReLUActivaction();
        }
    }

    class BinaryClassificationOutputLayer : Layer
    {
        public BinaryClassificationOutputLayer(int nodeCount) : base(nodeCount)
        {
            this.ActivationFunction = new SigmoidActivation();
        }
    }
}
