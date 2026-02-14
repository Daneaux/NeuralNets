using MatrixLibrary;
using NeuralNets;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace NeuralNetsTests.Loss
{
    [TestClass]
    public class LossTests
    {
        private const float Tolerance = 1e-4f;

        [TestMethod]
        [Ignore("TorchSharp API issue with single sample cross entropy")]
        public void CrossEntropyLoss_SingleSample_MatchesTorchSharp()
        {
            float[] logits = { 2.0f, 1.0f, 0.1f };
            float[] targetOneHot = { 1.0f, 0.0f, 0.0f };

            var truth = new AvxColumnVector(targetOneHot);
            var predicted = new AvxColumnVector(logits);

            var loss = new CategoricalCrossEntropy();
            float myLoss = loss.ScalarLoss(truth, predicted);

            float[,] logits2d = new float[1, 3] { { logits[0], logits[1], logits[2] } };
            float[,] target2d = new float[1, 3] { { targetOneHot[0], targetOneHot[1], targetOneHot[2] } };
            using var torchLogits = torch.from_array(logits2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLoss = functional.cross_entropy(torchLogits, torchTarget, reduction: Reduction.Sum);
            float torchLossValue = torchLoss.cpu().item<float>();

            Assert.AreEqual(torchLossValue, myLoss, Tolerance);
        }

        [TestMethod]
        public void CrossEntropyLoss_Batch_MatchesTorchSharp()
        {
            float[,] logits = { { 2.0f, 1.0f, 0.1f }, { 0.5f, 2.0f, 1.5f } };
            float[,] targetOneHot = { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } };

            var truth = MatrixFactory.CreateMatrix(targetOneHot);
            var predicted = MatrixFactory.CreateMatrix(logits);

            var loss = new CategoricalCrossEntropy();
            float myLoss = loss.ScalarLossBatch(truth, predicted);

            using var torchLogits = torch.from_array(logits);
            using var torchTarget = torch.from_array(new long[] { 0, 1 });
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);
            using var torchLoss = torchLossFn.forward(torchLogits, torchTarget);
            float torchLossValue = torchLoss.cpu().item<float>();

            Assert.AreEqual(torchLossValue, myLoss, Tolerance);
        }

        [TestMethod]
        public void CrossEntropyLoss_Gradient_MatchesTorchSharp()
        {
            float[] logits = { 2.0f, 1.0f, 0.1f };
            float[] targetOneHot = { 1.0f, 0.0f, 0.0f };

            var truth = new AvxColumnVector(targetOneHot);
            var predicted = new AvxColumnVector(logits);

            var loss = new CategoricalCrossEntropy();
            var myGradient = loss.Derivative(truth, predicted);

            using var torchLogits = torch.from_array(logits).reshape(1, 3).requires_grad_(true);
            using var torchTarget = torch.from_array(new long[] { 0 });
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);
            using var torchLoss = torchLossFn.forward(torchLogits, torchTarget);
            torchLoss.backward();

            var torchGradient = torchLogits.grad.cpu().data<float>().ToArray();

            for (int i = 0; i < logits.Length; i++)
            {
                Assert.AreEqual(torchGradient[i], myGradient[i], Tolerance, $"Gradient at index {i} does not match TorchSharp");
            }
        }

        [TestMethod]
        public void SparseCategoricalCrossEntropy_MatchesTorchSharp()
        {
            float[] logits = { 2.0f, 1.0f, 0.1f };
            long targetClass = 0;

            var truth = new AvxColumnVector(new float[] { targetClass });
            var predicted = new AvxColumnVector(logits);

            var loss = new SparseCategoricalCrossEntropy();
            float myLoss = loss.ScalarLoss(truth, predicted);

            using var torchLogits = torch.from_array(logits).reshape(1, 3);
            using var torchTarget = torch.from_array(new long[] { targetClass });
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);
            using var torchLoss = torchLossFn.forward(torchLogits, torchTarget);
            float torchLossValue = torchLoss.cpu().item<float>();

            Assert.AreEqual(torchLossValue, myLoss, Tolerance);
        }

        [TestMethod]
        public void CrossEntropyLoss_BatchGradient_MatchesTorchSharp()
        {
            float[,] logits = { { 2.0f, 1.0f, 0.1f }, { 0.5f, 2.0f, 1.5f } };
            long[] targetClasses = { 0, 1 };

            using var torchLogits = torch.from_array(logits).requires_grad_(true);
            using var torchTarget = torch.from_array(targetClasses);
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);
            using var torchLoss = torchLossFn.forward(torchLogits, torchTarget);
            torchLoss.backward();

            var torchGradient = torchLogits.grad.cpu().data<float>().ToArray();

            var loss = new CategoricalCrossEntropy();
            for (int sample = 0; sample < 2; sample++)
            {
                var truth = new AvxColumnVector(new float[] { 
                    targetClasses[sample] == 0 ? 1.0f : 0.0f,
                    targetClasses[sample] == 1 ? 1.0f : 0.0f,
                    targetClasses[sample] == 2 ? 1.0f : 0.0f
                });
                var predicted = new AvxColumnVector(new float[] { 
                    logits[sample, 0], logits[sample, 1], logits[sample, 2] 
                });

                var myGradient = loss.Derivative(truth, predicted);

                for (int i = 0; i < 3; i++)
                {
                    int idx = sample * 3 + i;
                    Assert.AreEqual(torchGradient[idx], myGradient[i], Tolerance, $"Gradient at sample {sample}, index {i} does not match TorchSharp");
                }
            }
        }
    }
}
