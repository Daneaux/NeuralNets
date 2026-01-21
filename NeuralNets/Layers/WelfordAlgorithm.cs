using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    public class WelfordAlgorithm
    {
        private long count;
        public double Mean { get; private set; }
        private double M2;
        
        public ColumnVectorBase Vec { get; }
        public double StandardDeviation { get; }
        public double StandardDeviation2 { get; }

        public WelfordAlgorithm(ColumnVectorBase vec)
        {
            count = 0;
            Mean = 0.0;
            M2 = 0.0;
            Vec = vec;

            foreach (float s in vec.Column)
                Update(s);

            StandardDeviation = this.GetStandardDeviation();
        }

        private void Update(double newValue)
        {
            count++;
            double delta = newValue - Mean;
            Mean += delta / count;
            double delta2 = newValue - Mean;
            M2 += delta * delta2;
        }

        public double GetVariance()
        {
            if (count < 2)
                return 0.0;
            return M2 / (count);
        }

        public double GetStandardDeviation()
        {
            return Math.Sqrt(GetVariance());
        }
    }

}
