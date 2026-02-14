using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace NeuralNetsTests.torchSharpComparison
{
    [TestClass]
    public class SquaredLossDebugTests
    {
        [TestMethod]
        public void Debug_OneTrainingStep_SquaredLoss()
        {
            int inputDim = 4;
            int hiddenDim = 3;
            float learningRate = 0.1f;

            // Fixed weights for reproducibility
            float[,] weights2D = new float[,] {
                { 0.1f, 0.2f, 0.3f, 0.4f },
                { 0.5f, 0.6f, 0.7f, 0.8f },
                { 0.9f, 1.0f, 1.1f, 1.2f }
            };
            float[] biasData = { 0.1f, 0.2f, 0.3f };

            float[] inputData = { 0.5f, 0.3f, 0.2f, 0.1f };
            float[] targetData = { 1.0f, 0.0f, 0.0f };

            Console.WriteLine("=== SQUARED LOSS TRAINING STEP DEBUG ===\n");

            // === TorchSharp ===
            // Clone arrays because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights = torch.from_array((float[,])weights2D.Clone());
            using var torchBiases = torch.from_array((float[])biasData.Clone());
            var torchLinear = Linear(inputDim, hiddenDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torchBiases.AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, hiddenDim];
            for (int i = 0; i < hiddenDim; i++) target2d[0, i] = targetData[i % 3];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = MSELoss(reduction: Reduction.Sum);

            var torchPred = torchLinear.forward(torchInput);
            var torchPredData = torchPred.cpu().data<float>().ToArray();
            var torchLoss = torchLossFn.forward(torchPred, torchTarget);
            torchLoss.backward();

            var torchInitialWeights = torchLinear.weight.clone().cpu().data<float>().ToArray();
            var torchGradW = torchLinear.weight.grad.cpu().data<float>().ToArray();
            var torchGradB = torchLinear.bias.grad.cpu().data<float>().ToArray();

            Console.WriteLine("TorchSharp:");
            Console.WriteLine($"  Predictions: [{string.Join(", ", torchPredData)}]");
            Console.WriteLine($"  Loss: {torchLoss.cpu().data<float>().ToArray()[0]:F6}");
            Console.WriteLine($"  Initial weights[0]: {torchInitialWeights[0]:F6}");
            Console.WriteLine($"  Gradients[0]: {torchGradW[0]:F6}");
            Console.WriteLine($"  Learning rate: {learningRate:F6}");
            Console.WriteLine($"  Expected weight update: -{torchGradW[0] * learningRate:F6}");
            Console.WriteLine($"  Expected new weight[0]: {torchInitialWeights[0] - torchGradW[0] * learningRate:F6}");

            // Manually update weights
            using (torch.no_grad())
            {
                var gradW = torchLinear.weight.grad;
                var gradB = torchLinear.bias.grad;
                torchLinear.weight.sub_(gradW * learningRate);
                torchLinear.bias.sub_(gradB * learningRate);
            }

            var torchUpdatedWeights = torchLinear.weight.cpu().data<float>().ToArray();
            Console.WriteLine($"  Actual updated weight[0]: {torchUpdatedWeights[0]:F6}\n");

            // === NeuralNets ===
            var weights = MatrixFactory.CreateMatrix(weights2D);
            var biases = new AvxColumnVector(biasData);
            
            // Debug: print the actual weights
            Console.WriteLine("NeuralNets weights:");
            for (int r = 0; r < 3; r++)
            {
                var row = new float[4];
                for (int c = 0; c < 4; c++) row[c] = weights[r, c];
                Console.WriteLine($"  Row {r}: [{string.Join(", ", row)}]");
            }
            Console.WriteLine($"  Biases: [{string.Join(", ", Enumerable.Range(0, biases.Size).Select(i => biases[i]))}]");
            
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var layer = new WeightedLayer(inputShape, hiddenDim, weights, biases);

            var layers = new System.Collections.Generic.List<Layer> { layer };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, hiddenDim, new MeanSquaredErrorLoss());

            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(targetData.Take(hiddenDim).ToArray());
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnOutput = layer.FeedFoward(nnInput.ToTensor());
            var nnOutputVec = nnOutput.ToColumnVector();

            Console.WriteLine("NeuralNets:");
            Console.WriteLine($"  Predictions: [{string.Join(", ", nnOutputVec!.Column)}]");
            Console.WriteLine($"  Initial weights[0]: {layer.Weights[0, 0]:F6}");

            // Backward pass
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, nnOutputVec);

            Console.WriteLine($"  Gradients[0]: {layer.LastWeightGradient[0, 0]:F6}");
            Console.WriteLine($"  Learning rate: {learningRate:F6}");
            Console.WriteLine($"  Expected weight update: -{layer.LastWeightGradient[0, 0] * learningRate:F6}");
            Console.WriteLine($"  Expected new weight[0]: {layer.Weights[0, 0] - layer.LastWeightGradient[0, 0] * learningRate:F6}");

            // Apply gradients
            layer.UpdateWeightsAndBiasesWithScaledGradients(learningRate);

            var nnUpdatedWeight0 = layer.Weights[0, 0];
            Console.WriteLine($"  Actual updated weight[0]: {nnUpdatedWeight0:F6}\n");

            // Compare
            Console.WriteLine("=== COMPARISON ===");
            Console.WriteLine($"TorchSharp weight[0]: {torchUpdatedWeights[0]:F6}");
            Console.WriteLine($"NeuralNets weight[0]: {nnUpdatedWeight0:F6}");
            Console.WriteLine($"Difference: {System.Math.Abs(torchUpdatedWeights[0] - nnUpdatedWeight0):F6}");

            // Show all weights for comparison
            Console.WriteLine("\nFull weight comparison:");
            for (int i = 0; i < torchUpdatedWeights.Length; i++)
            {
                int r = i / inputDim;
                int c = i % inputDim;
                float tw = torchUpdatedWeights[i];
                float nw = layer.Weights[r, c];
                float diff = System.Math.Abs(tw - nw);
                Console.WriteLine($"  [{r},{c}]: Torch={tw:F6}, NN={nw:F6}, Diff={diff:F6}");
            }

            // Assert
            for (int i = 0; i < torchUpdatedWeights.Length; i++)
            {
                int r = i / inputDim;
                int c = i % inputDim;
                Assert.AreEqual(torchUpdatedWeights[i], layer.Weights[r, c], 1e-3f,
                    $"Weight [{r},{c}] mismatch");
            }
        }
    }
}
