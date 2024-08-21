using System.Drawing;

namespace NumReaderNetwork
{

    /* USAGE
     * 
     * 
      foreach (var image in MnistReader.ReadTrainingData())
        {
            //use image here     
        }
        or

        foreach (var image in MnistReader.ReadTestData())
        {
            //use image here     
        }
    */

    public class Image
    {
        public byte Label { get; set; }
        public required byte[] Data { get; set; }
        public int Size { get { return this.Data.Length; } }
    }

    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

/*        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }*/
    }

    public static class MnistReader
    {
        private const string TrainImages = "mnistdataset\\train-images.idx3-ubyte";
        private const string TrainLabels = "mnistdataset\\train-labels.idx1-ubyte";
        private const string TestImages = "mnistdataset\\t10k-images.idx3-ubyte";
        private const string TestLabels = "mnistdataset\\t10k-labels.idx1-ubyte";

        public static IEnumerable<Image> ReadTrainingData()
        {
            foreach (var item in Read(TrainImages, TrainLabels))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData()
        {
            foreach (var item in Read(TestImages, TestLabels))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            imagesPath = Directory.GetCurrentDirectory() + "\\" + imagesPath;
            labelsPath = Directory.GetCurrentDirectory() + "\\" + labelsPath;
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                //var arr = new byte[height, width];

                //arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                yield return new Image()
                {
                    Data = bytes,
                    Label = labels.ReadByte()
                };
            }
        }
    }
}
