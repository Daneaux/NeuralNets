using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;
using MnistReader_ANN;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace NeuralNetsTests.MNIST
{
    [TestClass]
    public class MNISTComparisonTests
    {
        private const float Tolerance = 1e-4f;

        private static AvxColumnVector CreateColumnVector(float[] data) => new AvxColumnVector(data);

        private static TrainingPair CreateTrainingPair(float[] input, float[] output)
        {
            return new TrainingPair(CreateColumnVector(input).ToTensor(), CreateColumnVector(output).ToTensor());
        }

        [TestMethod]
        public void SimpleFeedForward_TorchSharpVsNeuralNets_SingleLayer()
        {
            int inputDim = 4;
            int outputDim = 3;
            float learningRate = 0.01f;

            float[] inputData = { 0.5f, 0.3f, 0.2f, 0.1f };
            float[] weightsData = { 
                0.1f, 0.2f, 0.3f, 0.4f,
                0.5f, 0.6f, 0.7f, 0.8f,
                0.9f, 1.0f, 1.1f, 1.2f
            };
            float[] biasData = { 0.1f, 0.2f, 0.3f };

            // === TorchSharp ===
            using var torchWeights = torch.from_array(weightsData, 3, 4);
            using var torchBias = torch.from_array(biasData);
            var torchLinear = Linear(inputDim, outputDim);
            torchLinear.weight.set_data(torchWeights);
            torchLinear.bias.set_data(torchBias);

            using var torchInput = torch.from_array(new float[,] { inputData });
            using var torchOutput = torchLinear.forward(torchInput);
            var torchOutputData = torchOutput.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights = MatrixFactory.CreateMatrix(new float[,] {
                { 0.1f, 0.2f, 0.3f, 0.4f },
                { 0.5f, 0.6f, 0.7f, 0.8f },
                { 0.9f, 1.0f, 1.1f, 1.2f }
            });
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var layer = new WeightedLayer(inputShape, outputDim, weights, biases);

            var nnInput = new AvxColumnVector(inputData);
            var nnOutput = layer.FeedFoward(nnInput.ToTensor());
            var nnOutputVec = nnOutput.ToColumnVector();
            var nnOutputData = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
                nnOutputData[i] = nnOutputVec[i];

            // Compare outputs
            for (int i = 0; i < outputDim; i++)
            {
                Assert.AreEqual(torchOutputData[i], nnOutputData[i], Tolerance, 
                    $"Output at index {i} does not match");
            }
        }

        [TestMethod]
        public void SimpleFeedForward_TorchSharpVsNeuralNets_MultiLayer()
        {
            int inputDim = 4;
            int hiddenDim = 3;
            int outputDim = 2;
            float learningRate = 0.01f;

            float[] inputData = { 0.5f, 0.3f, 0.2f, 0.1f };

            // Layer 1 weights (inputDim -> hiddenDim)
            float[] w1Data = { 
                0.1f, 0.2f, 0.3f, 0.4f,
                0.5f, 0.6f, 0.7f, 0.8f,
                0.9f, 1.0f, 1.1f, 1.2f
            };
            float[] b1Data = { 0.1f, 0.2f, 0.3f };

            // Layer 2 weights (hiddenDim -> outputDim)
            float[] w2Data = { 
                0.1f, 0.2f, 0.3f,
                0.4f, 0.5f, 0.6f
            };
            float[] b2Data = { 0.1f, 0.2f };

            // === TorchSharp ===
            var torchLinear1 = Linear(inputDim, hiddenDim);
            using var torchW1 = torch.from_array(w1Data, hiddenDim, inputDim);
            using var torchB1 = torch.from_array(b1Data);
            torchLinear1.weight.set_data(torchW1);
            torchLinear1.bias.set_data(torchB1);

            var torchLinear2 = Linear(hiddenDim, outputDim);
            using var torchW2 = torch.from_array(w2Data, outputDim, hiddenDim);
            using var torchB2 = torch.from_array(b2Data);
            torchLinear2.weight.set_data(torchW2);
            torchLinear2.bias.set_data(torchB2);

            using var torchInput = torch.from_array(new float[,] { inputData });
            var torchHidden = relu(torchLinear1.forward(torchInput));
            var torchOutput = torchLinear2.forward(torchHidden);
            var torchHiddenData = torchHidden.cpu().data<float>().ToArray();
            var torchOutputData = torchOutput.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);

            var weights1 = MatrixFactory.CreateMatrix(new float[,] {
                { 0.1f, 0.2f, 0.3f, 0.4f },
                { 0.5f, 0.6f, 0.7f, 0.8f },
                { 0.9f, 1.0f, 1.1f, 1.2f }
            });
            var biases1 = new AvxColumnVector(b1Data);
            var layer1 = new WeightedLayer(inputShape, hiddenDim, weights1, biases1);
            var relu1 = new ReLUActivation();

            var hiddenShape = layer1.OutputShape;
            var weights2 = MatrixFactory.CreateMatrix(new float[,] {
                { 0.1f, 0.2f, 0.3f },
                { 0.4f, 0.5f, 0.6f }
            });
            var biases2 = new AvxColumnVector(b2Data);
            var layer2 = new WeightedLayer(hiddenShape, outputDim, weights2, biases2);

            var layers = new List<Layer> { layer1, relu1, layer2 };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new SquaredLoss());

            var nnInput = new AvxColumnVector(inputData);
            var nnOutput = network.FeedForward(nnInput.ToTensor());
            var nnOutputVec = nnOutput.ToColumnVector();
            var nnOutputData = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
                nnOutputData[i] = nnOutputVec[i];

            // Compare outputs
            for (int i = 0; i < outputDim; i++)
            {
                Assert.AreEqual(torchOutputData[i], nnOutputData[i], Tolerance,
                    $"Output at index {i} does not match");
            }
        }

        [TestMethod]
        public void TrainingComparison_TorchSharpVsNeuralNets_OneStep()
        {
            int inputDim = 4;
            int outputDim = 3;
            float learningRate = 0.1f;

            float[] inputData = { 0.5f, 0.3f, 0.2f, 0.1f };
            float[] targetData = { 1.0f, 0.0f, 0.0f };

            float[] weightsData = { 
                0.1f, 0.2f, 0.3f, 0.4f,
                0.5f, 0.6f, 0.7f, 0.8f,
                0.9f, 1.0f, 1.1f, 1.2f
            };
            float[] biasData = { 0.1f, 0.2f, 0.3f };

            // === TorchSharp ===
            var torchLinear = Linear(inputDim, outputDim);
            using var torchWeights = torch.from_array(weightsData, outputDim, inputDim);
            using var torchBias = torch.from_array(biasData);
            torchLinear.weight.set_data(torchWeights.clone());
            torchLinear.bias.set_data(torchBias.clone());

            using var torchInput = torch.from_array(new float[,] { inputData });
            using var torchTarget = torch.from_array(new float[,] { targetData });
            using var torchLossFn = MSELoss();

            var torchPred = torchLinear.forward(torchInput);
            var torchLoss = torchLossFn.forward(torchPred, torchTarget);
            torchLoss.backward();

            var torchGradW = torchLinear.weight.grad.cpu().data<float>().ToArray();
            var torchGradB = torchLinear.bias.grad.cpu().data<float>().ToArray();

            // Store original weights for comparison
            var torchOrigWeights = torchWeights.clone();
            var torchOrigBias = torchBias.clone();

            // Update weights (manual SGD step)
            using (no_grad())
            {
                torchLinear.weight.set_(torchOrigWeights - learningRate * torchLinear.weight.grad);
                torchLinear.bias.set_(torchOrigBias - learningRate * torchLinear.bias.grad);
            }

            var torchNewWeights = torchLinear.weight.detach().cpu().data<float>().ToArray();
            var torchNewBiases = torchLinear.bias.detach().cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights = MatrixFactory.CreateMatrix(new float[,] {
                { 0.1f, 0.2f, 0.3f, 0.4f },
                { 0.5f, 0.6f, 0.7f, 0.8f },
                { 0.9f, 1.0f, 1.1f, 1.2f }
            });
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var layer = new WeightedLayer(inputShape, outputDim, weights, biases);

            var layers = new List<Layer> { layer };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new SquaredLoss());

            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(targetData);
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnOutput = network.FeedForward(nnInput.ToTensor());

            // Get loss
            var lossVec = network.GetLossVector(trainingPair, nnOutput.ToColumnVector());

            // Backward pass - use RenderContext
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, nnOutput.ToColumnVector());

            // Get gradients
            var nnGradW = new float[outputDim * inputDim];
            var nnWeightGrad = layer.WeightGradientAccumulator;
            for (int r = 0; r < outputDim; r++)
                for (int c = 0; c < inputDim; c++)
                    nnGradW[r * inputDim + c] = nnWeightGrad[r, c];

            var nnGradB = new float[outputDim];
            var nnBiasGrad = layer.BiasGradientAccumulator;
            for (int i = 0; i < outputDim; i++)
                nnGradB[i] = nnBiasGrad[i];

            // Compare gradients
            for (int i = 0; i < torchGradW.Length; i++)
            {
                Assert.AreEqual(torchGradW[i], nnGradW[i], Tolerance * 10,
                    $"Weight gradient at index {i} does not match");
            }
            for (int i = 0; i < torchGradB.Length; i++)
            {
                Assert.AreEqual(torchGradB[i], nnGradB[i], Tolerance * 10,
                    $"Bias gradient at index {i} does not match");
            }
        }

        [TestMethod]
        public void TrainingComparison_TorchSharpVsNeuralNets_WithSoftmax()
        {
            int inputDim = 4;
            int outputDim = 3;
            float learningRate = 0.1f;

            float[] inputData = { 0.5f, 0.3f, 0.2f, 0.1f };
            float[] targetData = { 1.0f, 0.0f, 0.0f };

            float[] weightsData = { 
                0.1f, 0.2f, 0.3f, 0.4f,
                0.5f, 0.6f, 0.7f, 0.8f,
                0.9f, 1.0f, 1.1f, 1.2f
            };
            float[] biasData = { 0.1f, 0.2f, 0.3f };

            // === TorchSharp ===
            var torchLinear = Linear(inputDim, outputDim);
            using var torchWeights = torch.from_array(weightsData, outputDim, inputDim);
            using var torchBias = torch.from_array(biasData);
            torchLinear.weight.set_data(torchWeights.clone());
            torchLinear.bias.set_data(torchBias.clone());

            using var torchInput = torch.from_array(new float[,] { inputData });
            using var torchTarget = torch.tensor(new long[] { 0 });
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);

            var torchPred = torchLinear.forward(torchInput);
            var torchLoss = torchLossFn.forward(torchPred, torchTarget);
            torchLoss.backward();

            var torchGradW = torchLinear.weight.grad.cpu().data<float>().ToArray();
            var torchGradB = torchLinear.bias.grad.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights = MatrixFactory.CreateMatrix(new float[,] {
                { 0.1f, 0.2f, 0.3f, 0.4f },
                { 0.5f, 0.6f, 0.7f, 0.8f },
                { 0.9f, 1.0f, 1.1f, 1.2f }
            });
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var layer = new WeightedLayer(inputShape, outputDim, weights, biases);

            var layers = new List<Layer> { layer };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(new float[] { 0 }); // sparse target
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnOutput = network.FeedForward(nnInput.ToTensor());

            // Get loss
            var lossVec = network.GetLossVector(trainingPair, nnOutput.ToColumnVector());

            // Backward pass
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, nnOutput.ToColumnVector());

            // Get gradients
            var nnGradW = new float[outputDim * inputDim];
            var nnWeightGrad = layer.WeightGradientAccumulator;
            for (int r = 0; r < outputDim; r++)
                for (int c = 0; c < inputDim; c++)
                    nnGradW[r * inputDim + c] = nnWeightGrad[r, c];

            var nnGradB = new float[outputDim];
            var nnBiasGrad = layer.BiasGradientAccumulator;
            for (int i = 0; i < outputDim; i++)
                nnBiasGrad[i] = nnBiasGrad[i];

            // Compare gradients - with higher tolerance due to numerical differences
            for (int i = 0; i < torchGradW.Length; i++)
            {
                Assert.AreEqual(torchGradW[i], nnGradW[i], 0.01f,
                    $"Weight gradient at index {i} does not match");
            }
        }
    }
}
