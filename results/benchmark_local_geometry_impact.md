# LocalGeometry Performance Benchmark Results

## Execution Time Results

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
| extracted j | 17.02 | 6.2% |
| full lg multiple | 18.87 | 17.8% |
| baseline simple | 16.02 | 0.0% |
| full lg jacobian | 17.15 | 7.1% |
| simplified lg | 16.44 | 2.6% |

## Memory Footprint

- Full LocalGeometry: 0.0025634765625 MB (21.0x scalar field)
- Extracted J: 0.0001220703125 MB (1.0x scalar field)

## Key Finding

Full LocalGeometry overhead: 7.1%

⚠️ **MODERATE OVERHEAD** - Consider optimization strategies
