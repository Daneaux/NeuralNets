using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public class Layer
    {
        public int NumNodes { get; private set; }

        protected Layer(int nodeCount)
        {
            this.NumNodes = nodeCount;
        }
    }
    
    public class WeightedLayer : Layer
    {
        public IActivationFunction ActivationFunction { get; protected set; }
        public WeightedLayer(int nodeCount, IActivationFunction activationFunction, int incomingDataPoints) : base(nodeCount)
        {
            this.ActivationFunction = activationFunction;
            this.Weights = new Matrix(nodeCount, incomingDataPoints);
            this.Weights.SetRandom(123, - Math.Sqrt(nodeCount), Math.Sqrt(nodeCount)); // Xavier initilization
            this.Biases = new ColumnVector(nodeCount);
            this.Biases.SetRandom(123, -0.01, 0.01);            
        }

        // Weight matrix is for one sample, and the number of rows corresponds to the number of hidden layer nodes, for example 16.
        // And the number of columns is the number of data points in a samples, for examle 768 b&w pixel values for the MNIST number set
        public Matrix Weights { get; private set; }
        public ColumnVector Biases { get; private set; }
    }

    public class InputLayer : Layer 
    { 
        public InputLayer(int nodeCount) : base(nodeCount)
        {
        }

        public InputLayer(ColumnVector valueVector) : base(valueVector.Size)
        {
            ValueVector = valueVector;
        }

        public ColumnVector ValueVector { get; internal set; }
    }

}
