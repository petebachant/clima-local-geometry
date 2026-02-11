# LocalGeometry Performance Benchmark Results

## Execution Time Results

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
| extracted j | 16.82 | 5.3% |
| full lg multiple | 18.91 | 18.4% |
| baseline simple | 15.97 | 0.0% |
| full lg jacobian | 17.10 | 7.1% |
| simplified lg | 16.70 | 4.6% |

## Memory Footprint

- Full LocalGeometry: 0.0025634765625 MB (21.0x scalar field)
- Extracted J: 0.0001220703125 MB (1.0x scalar field)

## Key Finding

Full LocalGeometry overhead: 7.1%

⚠️ **MODERATE OVERHEAD** - Consider optimization strategies
