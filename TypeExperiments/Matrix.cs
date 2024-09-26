using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TypeExperiments
{
    public class Matrix1
    {
    }

    public class Matrix2
    {
    }

    public class Vector1
    {

    }

    public class Vector2
    {
    }

    public abstract class Tensor
    {

    }

    public class ConvTensor : Tensor
    {
        public List<Matrix1> matrix1s;
    }

    public class AnnTensor : Tensor
    {
        public Matrix1 matrix1;
    }
}
