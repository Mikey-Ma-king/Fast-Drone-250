# Extrinsic experiment summary (n=5 repeats per cell)


## Exp A: Jump — IAE pre-convergence (m·s)
| Method | wavy_circle | sin_accel_straight |
|--------|-------------|-------------------|
| Ours (cascade) | 0.62±0.58 | 0.84±0.97 |
| Whole KF | 58.74±6.09 | 38.06±6.19 |
| LKF incr | 94.50±5.94 | 60.03±20.19 |
| Vis fixed R/t | 10.93±12.96 | 6.36±1.11 |

## Exp A: Jump — convergence time (s)
| Method | wavy_circle | sin_accel_straight |
|--------|-------------|-------------------|
| Ours (cascade) | 1.44±0.70 | 2.27±2.52 |
| Whole KF | 31.59±0.73 | 30.92±4.91 |
| LKF incr | NC (0/5) | 10.65 (1/5) |
| Vis fixed R/t | 12.29±16.40 (4/5) | 8.96±1.43 |

## Exp B: Drift — mean error (m, ours/kf post-conv)
| Method | wavy_circle | sin_accel_straight |
|--------|-------------|-------------------|
| Ours (cascade) | 0.127±0.004 | 0.122±0.016 |
| Whole KF | 0.139±0.012 | 0.109±0.005 |
| LKF incr | 1.360±0.084 | 0.620±0.066 |
| Vis fixed R/t | 0.518±0.016 | 0.557±0.020 |

## Exp B: Drift — max error (m, ours/kf post-conv)
| Method | wavy_circle | sin_accel_straight |
|--------|-------------|-------------------|
| Ours (cascade) | 0.307±0.025 | 0.294±0.034 |
| Whole KF | 0.336±0.025 | 0.284±0.051 |
| LKF incr | 2.549±0.420 | 1.580±0.264 |
| Vis fixed R/t | 1.313±0.053 | 2.123±0.149 |

## Exp B: Drift — min error (m)
| Method | wavy_circle | sin_accel_straight |
|--------|-------------|-------------------|
| Ours (cascade) | 0.011±0.005 | 0.011±0.007 |
| Whole KF | 0.016±0.008 | 0.009±0.003 |
| LKF incr | 0.648±0.046 | 0.046±0.020 |
| Vis fixed R/t | 0.091±0.025 | 0.043±0.021 |
