


1.13
(b)
result on L4
Batch size: 4, Context length: 256
Warmup rounds: 5

Benchmarking small model...
  Forward:  0.0228 ± 0.0014 seconds
  Backward: 0.0293 ± 0.0020 seconds
  Total:    0.0521 seconds

Benchmarking medium model...


  Forward:  0.0529 ± 0.0003 seconds
  Backward: 0.0912 ± 0.0019 seconds
  Total:    0.1441 seconds

Benchmarking large model...
  Forward:  0.1291 ± 0.0033 seconds
  Backward: 0.2032 ± 0.0010 seconds
  Total:    0.3323 seconds

Benchmarking xl model...
  Forward:  0.2696 ± 0.0064 seconds
  Backward: 0.4150 ± 0.0034 seconds
  Total:    0.6847 seconds

Benchmarking 2.7B model...
  Forward:  0.3769 ± 0.0063 seconds
  Backward: 0.6210 ± 0.0015 seconds
  Total:    0.9979 seconds


Summary Results:
================================================================================
| Model   |   d_model |   d_ff |   num_layers |   num_heads |   vocab_size |   context_length |   batch_size |   Forward Mean (s) |   Forward Std (s) |   Backward Mean (s) |   Backward Std (s) |   Total Mean (s) |   Total Std (s) |
|:--------|----------:|-------:|-------------:|------------:|-------------:|-----------------:|-------------:|-------------------:|------------------:|--------------------:|-------------------:|-----------------:|----------------:|
| small   |       768 |   3072 |           12 |          12 |        10000 |              256 |            4 |             0.0228 |            0.0014 |              0.0293 |             0.0020 |           0.0521 |   0.0024 |
| medium  |      1024 |   4096 |           24 |          16 |        10000 |              256 |            4 |             0.0529 |            0.0003 |              0.0912 |             0.0019 |           0.1441 |   0.0019 |
| large   |      1280 |   5120 |           36 |          20 |        10000 |              256 |            4 |             0.1291 |            0.0033 |              0.2032 |             0.0010 |           0.3323 |   0.0034 |
| xl      |      1600 |   6400 |           48 |          25 |        10000 |              256 |            4 |             0.2696 |            0.0064 |              0.4150 |             0.0034 |           0.6847 |   0.0072 |
| 2.7B    |      2560 |  10240 |           32 |          32 |        10000 |              256 |            4 |             0.3769 |            0.0063 |              0.6210 |             0.0015 |           0.9979 |   0.0065 |

Running benchmarks on cuda
Batch size: 4, Context length: 256
Warmup rounds: 0

Benchmarking small model...
  Forward:  0.1808 ± 0.4705 seconds
  Backward: 0.0689 ± 0.1221 seconds
  Total:    0.2497 seconds

Benchmarking medium model...
  Forward:  0.0557 ± 0.0140 seconds
  Backward: 0.0871 ± 0.0036 seconds
  Total:    0.1429 seconds

Benchmarking large model...
  Forward:  0.1290 ± 0.0102 seconds
  Backward: 0.1965 ± 0.0071 seconds
  Total:    0.3255 seconds

Benchmarking xl model...
  Forward:  0.2610 ± 0.0071 seconds
  Backward: 0.4002 ± 0.0030 seconds
  Total:    0.6612 seconds

Benchmarking 2.7B model...
  Forward:  0.3659 ± 0.0135 seconds
  Backward: 0.6013 ± 0.0022 seconds
  Total:    0.9672 seconds


Summary Results:
================================================================================
| Model   |   d_model |   d_ff |   num_layers |   num_heads |   vocab_size |   context_length |   batch_size |   Forward Mean (s) |   Forward Std (s) |   Backward Mean (s) |   Backward Std (s) |   Total Mean (s) |   Total Std (s) |
|:--------|----------:|-------:|-------------:|------------:|-------------:|-----------------:|-------------:|-------------------:|------------------:|--------------------:|-------------------:|-----------------:|----------------:|
| small   |       768 |   3072 |           12 |          12 |        10000 |              256 |            4 |             0.1808 |            0.4705 |              0.0689 |             0.1221 |           0.2497 |   0.4860 |
| medium  |      1024 |   4096 |           24 |          16 |        10000 |              256 |            4 |             0.0557 |            0.0140 |              0.0871 |             0.0036 |           0.1429 |   0.0144 |
| large   |      1280 |   5120 |           36 |          20 |        10000 |              256 |            4 |             0.1290 |            0.0102 |              0.1965 |             0.0071 |           0.3255 |   0.0124 |
| xl      |      1600 |   6400 |           48 |          25 |        10000 |              256 |            4 |             0.2610 |            0.0071 |              0.4002 |             0.0030 |           0.6612 |   0.0077 |
| 2.7B    |      2560 |  10240 |           32 |          32 |        10000 |              256 |            4 |             0.3659 |            0.0135 |              0.6013 |             0.0022 |           0.9672 |   0.0137 |

on H100:
Running benchmarks on cuda
Batch size: 4, Context length: 256
Warmup rounds: 5

Benchmarking small model...
  Forward:  0.0312 ± 0.0002 seconds
  Backward: 0.0555 ± 0.0013 seconds
  Total:    0.0867 seconds

Benchmarking medium model...
  Forward:  0.0634 ± 0.0001 seconds
  Backward: 0.0983 ± 0.0005 seconds
  Total:    0.1617 seconds

Benchmarking large model...
  Forward:  0.0967 ± 0.0009 seconds
  Backward: 0.1853 ± 0.0009 seconds
  Total:    0.2820 seconds

Benchmarking xl model...
  Forward:  0.1649 ± 0.0033 seconds
  Backward: 0.3450 ± 0.0023 seconds
  Total:    0.5099 seconds

Benchmarking 2.7B model...
  Forward:  0.2525 ± 0.0053 seconds
  Backward: 0.5222 ± 0.0014 seconds
  Total:    0.7746 seconds


Summary Results:
================================================================================
| Model   |   d_model |   d_ff |   num_layers |   num_heads |   vocab_size |   context_length |   batch_size |   Forward Mean (s) |   Forward Std (s) |   Backward Mean (s) |   Backward Std (s) |   Total Mean (s) |   Total Std (s) |
|:--------|----------:|-------:|-------------:|------------:|-------------:|-----------------:|-------------:|-------------------:|------------------:|--------------------:|-------------------:|-----------------:|----------------:|
| small   |       768 |   3072 |           12 |          12 |        10000 |              256 |            4 |             0.0312 |            0.0002 |              0.0555 |             0.0013 |           0.0867 |   0.0013 |
| medium  |      1024 |   4096 |           24 |          16 |        10000 |              256 |            4 |             0.0634 |            0.0001 |              0.0983 |             0.0005 |           0.1617 |   0.0005 |
| large   |      1280 |   5120 |           36 |          20 |        10000 |              256 |            4 |             0.0967 |            0.0009 |              0.1853 |             0.0009 |           0.2820 |   0.0013 |
| xl      |      1600 |   6400 |           48 |          25 |        10000 |              256 |            4 |             0.1649 |            0.0033 |              0.3450 |             0.0023 |           0.5099 |   0.0040 |
| 2.7B    |      2560 |  10240 |           32 |          32 |        10000 |              256 |            4 |             0.2525 |            0.0053 |              0.5222 |             0.0014 |           0.7746 |   0.0055 |


