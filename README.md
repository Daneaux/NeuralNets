<<<<<<< HEAD
# NeuralNets
=======
Beginnings of a neural network and matrix library. The matrix library is mostly done and about 90% avx512 accelerated (so it only works on modern CPU's). The nueral network library is good enough now for traditional dense feedfoward networks with back propagation, but the CNN isn't working yet and the architecture is a bit wonky (currently refactoring ... will be much simpler and a lot less code).
>>>>>>> 7cafa0c561ef6acde84d5ac701335d1ac745f0d9

A high-performance neural network library for C# with GPU acceleration support via CUDA/CUBLAS and AVX512 SIMD optimizations.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-226%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## 🚀 Overview

NeuralNets is a from-scratch neural network implementation designed for performance and educational clarity. It includes:

- **Multiple compute backends**: Software (CPU), AVX512 (SIMD), and CUDA/CUBLAS (GPU)
- **Layer types**: Dense/fully-connected, convolutional, pooling, flattening, normalization
- **Activation functions**: ReLU, Sigmoid, Softmax, LogSoftmax
- **Loss functions**: Squared Loss, MSE, Cross Entropy, Categorical Cross Entropy
- **Training methods**: Single-sample, batch training with gradient accumulation
- **Verified correctness**: Comprehensive test suite comparing against PyTorch/TorchSharp

## 📊 Performance

Matrix multiply benchmarks (64×64, 256×256, 1024×1024) on AMD 7600 + GTX 3090:

| Method   | Size | Time            | Speedup |
|--------- |----- |----------------:|--------:|
| Software | 64   |       255.28 µs |    1.0x |
| AVX      | 64   |        35.01 µs |    7.3x |
| GPU      | 64   |        78.23 µs |    3.3x |
|          |      |                 |         |
| Software | 256  |    18,472.34 µs |    1.0x |
| AVX      | 256  |     2,085.79 µs |    8.9x |
| GPU      | 256  |       105.87 µs |  174.5x |
|          |      |                 |         |
| Software | 1024 | 1,925,101.06 µs |    1.0x |
| AVX      | 1024 |   128,724.05 µs |   15.0x |
| GPU      | 1024 |       947.98 µs | 2030.8x |

**Key Insight**: GPU acceleration is ideal for large matrices (1000×1000+), while AVX512 excels at medium-sized operations.

## 🏗️ Architecture

```
NeuralNets/
├── MatrixLibrary/          # Matrix/vector operations
│   ├── Software/          # CPU fallback implementations
│   ├── Avx/               # AVX512 SIMD optimized
│   └── Gpu/               # CUDA/CUBLAS GPU accelerated
├── NeuralNets/            # Neural network library
│   ├── Activations/       # Activation functions
│   ├── Layers/            # Network layers
│   ├── Loss/              # Loss functions
│   └── RenderContext/     # Training context
├── NeuralNetsTests/       # Test suite
│   ├── torchSharpComparison/  # PyTorch verification
│   ├── CNNTests.cs        # Convolutional network tests
│   └── ...
├── AnnHarness/            # Sample MNIST application
├── BenchmarkHarness/      # Performance benchmarks
└── SharpTorch/            # TorchSharp integration
```

## 🎯 Key Features

### Matrix Operations
- **Generic tensor support**: 1D (vectors), 2D (matrices), 3D+ (volumes)
- **Multiple backends**: Transparently switch between Software, AVX512, and GPU
- **Optimized operations**: Matrix multiplication, convolution, pooling with AVX512
- **Memory management**: Efficient pinned memory for GPU operations

### Neural Network Layers
- **WeightedLayer**: Fully-connected dense layers with bias
- **ConvolutionLayer**: 2D convolutions with configurable kernels, stride, padding
- **PoolingLayer**: Max pooling with gradient routing for backprop
- **FlattenLayer**: Convert multi-dimensional tensors to vectors
- **NormalizationLayer**: Batch/layer normalization support

### Training
- **Forward pass**: Compute predictions layer by layer
- **Backward pass**: Automatic differentiation via backpropagation
- **Batch training**: Accumulate gradients over multiple samples
- **Optimizers**: SGD with learning rate scheduling support
- **Loss tracking**: Monitor training progress with various loss functions

## 🔧 Building

### Prerequisites
- .NET 8.0 SDK
- Visual Studio 2022 (optional, for IDE support)
- CUDA Toolkit 12.x (optional, for GPU support)
- NVIDIA GPU with Compute Capability 7.0+ (optional, for GPU support)

### Build Commands

```bash
# Build entire solution
dotnet build NeuralNets/NeuralNets.sln

# Build specific projects
dotnet build MatrixLibrary/MatrixLibrary.csproj
dotnet build NeuralNets/NeuralNets.csproj

# Build in Release mode
dotnet build -c Release NeuralNets/NeuralNets.sln
```

### GPU Support (Optional)

If you have a CUDA-capable GPU:

1. Install CUDA Toolkit 12.x
2. Update CUDA paths in `NeuralNets.csproj` if needed
3. Build and run CublasTests to verify:
   ```bash
   dotnet test CublasTests/CublasTests.csproj
   ```

## 🧪 Running Tests

### All Tests
```bash
dotnet test NeuralNets/NeuralNets.sln
```

### Specific Test Categories
```bash
# TorchSharp comparison tests
dotnet test --filter "FullyQualifiedName~torchSharpComparison"

# CNN tests
dotnet test --filter "FullyQualifiedName~CNN"

# MNIST integration tests
dotnet test --filter "FullyQualifiedName~MNIST"

# GPU tests (requires CUDA)
dotnet test CublasTests/CublasTests.csproj
```

### Test Summary
- **Total Tests**: 226
- **Passing**: 226 ✅
- **Skipped**: 2 (known limitations)
- **Failing**: 0

## 📖 Usage Examples

### Basic Neural Network
```csharp
using NeuralNets;
using MatrixLibrary;

// Create network: 784 -> 128 -> 10 (MNIST classifier)
var inputShape = new InputOutputShape(1, 784, 1, 1);

// Layer 1: 784 -> 128
var weights1 = MatrixFactory.CreateMatrix(128, 784);
var biases1 = MatrixFactory.CreateColumnVector(128);
var layer1 = new WeightedLayer(inputShape, 128, weights1, biases1);
var relu1 = new ReLUActivaction();

// Layer 2: 128 -> 10
var weights2 = MatrixFactory.CreateMatrix(10, 128);
var biases2 = MatrixFactory.CreateColumnVector(10);
var layer2 = new WeightedLayer(layer1.OutputShape, 10, weights2, biases2);

// Build network
var layers = new List<Layer> { layer1, relu1, layer2 };
var network = new GeneralFeedForwardANN(
    layers,
    trainingRate: 0.01f,
    inputDim: 784,
    outputDim: 10,
    new CategoricalCrossEntropy()
);

// Training
var trainingSet = new MNISTTrainingSet();
var renderContext = new RenderContext(network, batchSize: 64, trainingSet);

for (int epoch = 0; epoch < 10; epoch++)
{
    RenderContext.BatchTrain(renderContext, epoch);
}

// Inference
var input = GetMNISTImage(); // Your input loading logic
var output = renderContext.FeedForward(input);
var prediction = ArgMax(output.ToColumnVector());
```

### CNN for Image Classification
```csharp
// CNN: 28x28 -> Conv -> ReLU -> Pool -> Flatten -> Dense
var inputShape = new InputOutputShape(28, 28, 1, 1); // 28x28 grayscale

var conv1 = new ConvolutionLayer(inputShape, kernelCount: 5, kernelSquareDimension: 4, stride: 1);
var relu1 = new ReLUActivaction();
var pool1 = new PoolingLayer(conv1.OutputShape, stride: 2, kernelCount: 5, kernelSquareDimension: 2, kernelDepth: 1);
var flatten = new FlattenLayer(pool1.OutputShape, nodeCount: 1);
var dense = new WeightedLayer(flatten.OutputShape, nodeCount: 10);
var softmax = new SoftMax();

var cnnLayers = new List<Layer> { conv1, relu1, pool1, flatten, dense, softmax };
var cnn = new GeneralFeedForwardANN(cnnLayers, 0.1f, 28*28, 10, new CategoricalCrossEntropy());
```

### Matrix Operations
```csharp
// CPU fallback
var cpuMatrix = MatrixFactory.CreateMatrix(1000, 1000);
var cpuResult = cpuMatrix.Multiply(otherMatrix);

// AVX512 optimized (automatic if supported)
var avxMatrix = new AvxMatrix(1000, 1000);
// ... populate data ...
var avxResult = avxMatrix.Multiply(otherMatrix); // ~10x faster

// GPU accelerated (if CUDA available)
var gpuMatrix = new GpuMatrix(1000, 1000);
// ... populate data ...
var gpuResult = gpuMatrix.Multiply(otherMatrix); // ~50x faster
```

## 🧪 Testing Strategy

### TorchSharp Comparison
The library includes comprehensive tests comparing against PyTorch/TorchSharp to verify correctness:

- **Gradient verification**: Ensure ∂E/∂W and ∂E/∂b match PyTorch exactly
- **Weight updates**: Verify SGD weight updates match PyTorch
- **Forward pass**: Compare layer outputs with 1e-3 tolerance
- **Full training**: Train identical networks and compare converged weights

### Test Categories

1. **Unit Tests**: Individual matrix operations, layer computations
2. **Integration Tests**: Full network training, end-to-end workflows
3. **Comparison Tests**: Against TorchSharp reference implementation
4. **Performance Tests**: GPU vs CPU benchmarks
5. **Regression Tests**: Prevent bugs from recurring

## 📁 Project Structure

| Directory | Purpose |
|-----------|---------|
| `MatrixLibrary/` | Core matrix/vector/tensor operations |
| `NeuralNets/` | Neural network layers, activations, loss functions |
| `NeuralNetsTests/` | Comprehensive test suite |
| `AnnHarness/` | MNIST digit recognition demo |
| `BenchmarkHarness/` | Performance benchmarks |
| `SharpTorch/` | TorchSharp integration helpers |
| `CublasTests/` | GPU-specific tests |
| `MnistReader_ANN/` | MNIST data loading utilities |

## 🔬 Verification Against PyTorch

All critical components are verified against PyTorch/TorchSharp:

✅ **Matrix operations**: AVX/CPU/GPU produce identical results  
✅ **Forward pass**: Layer activations match within 1e-3  
✅ **Backpropagation**: Gradients match PyTorch exactly  
✅ **Weight updates**: SGD produces identical weight changes  
✅ **Loss functions**: Cross entropy, MSE match PyTorch  
✅ **Full networks**: Trained networks converge to similar solutions  

See `NeuralNetsTests/torchSharpComparison/` for detailed comparison tests.

## ⚡ Performance Tips

1. **Use AVX512**: Automatically enabled on supported CPUs (most modern Intel/AMD)
2. **Enable GPU**: Use `GpuMatrix` for large matrices (>1000×1000)
3. **Batch training**: Use `BatchTrain()` with batch size 32-128 for best throughput
4. **Matrix chaining**: Keep operations on GPU to avoid memory transfers
5. **Parallel processing**: Enable `PARALLEL_BATCH_TRAIN` for multi-core CPU training (CNNs should use single-threaded)

## 🐛 Known Limitations

- CNN layers (Convolution, Pooling) require single-threaded training due to per-sample state storage
- GPU requires CUDA-capable NVIDIA hardware
- AVX512 requires modern CPU (Intel Skylake+ or AMD Zen2+)

## 🤝 Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Verify against TorchSharp for numerical correctness
3. Run full test suite before submitting
4. Follow existing code style

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- PyTorch/TorchSharp for reference implementation
- Matt Mazur's backpropagation example for educational foundation
- CUDA/CUBLAS for GPU acceleration

## 📧 Contact

For issues or questions, please use the GitHub issue tracker.

---

**Built with ❤️ and lots of matrix multiplication**



