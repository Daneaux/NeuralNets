using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TypeExperiments
{

    public class ConvLayer : Layer
    {
        public override Tensor FeedForward(Tensor x)
        {
            Tensor tensor = new ConvTensor();
            return tensor;
        }


    }

    public abstract class Layer
    {
        public abstract Tensor FeedForward(Tensor x);
    }

    public class WeightLayer : Layer
    {
        public override Tensor FeedForward(Tensor x)
        {
            if(x is AnnTensor)
            {
                var y = (AnnTensor)x;
                var m = y.matrix1;
                return y;
            }
            return null;
        }
    }




}
