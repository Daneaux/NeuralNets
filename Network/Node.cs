using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    class Node
    {
        public int OutEdges => OutWeights.Length;
        public double[] OutWeights { get;  set; }
        public Node(int outEdges) 
        {
            OutWeights = new double[outEdges];
        }

        public void InitWeights(int seed, double min, double max)
        {
            Random r = new Random(seed);
            double width = max - min;
            for (int i = 0; i < OutWeights.Length; i++)
            {
                OutWeights[i] = (r.NextDouble() * width) + min;
            }
        }

        public IActivationFunction ActivationFunction { get; private set; }

        public double Activate(double weightedSumInput)
        {
            double output = ActivationFunction.Activate(weightedSumInput);
            for(int i=0; i<OutWeights.Length; i++)
            {
                OutWeights[i] *= output;
            }
            return output;
        }


    }
}
