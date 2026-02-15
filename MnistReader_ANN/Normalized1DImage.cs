
namespace MnistReader_ANN
{
    public class Normalized1DImage : Image
    {
        public required float[] Data { get; set; }
        public override int Size { get { return Data.Length; } }

    }
}
