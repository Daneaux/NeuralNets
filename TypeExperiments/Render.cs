using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TypeExperiments
{
    public class Render
    {
        public List<Layer> layers;

        public void DoRender()
        {
            Tensor tt = new ConvTensor();

            foreach (var layer in layers)
            {
                tt = layer.FeedForward(tt);
            }
        }
    }


}
