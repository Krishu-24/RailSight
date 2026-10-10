| variant | frames | frames_with_obstacle | precision | recall | F1 | false_alarm_rate | object_recall |
|---|---|---|---|---|---|---|---|
| class only | 836 | 102 | 0.590 | 0.480 | 0.530 | 0.046 | 0.445 |
| class + conf<0.3 | 836 | 102 | 0.586 | 0.500 | 0.540 | 0.049 | 0.445 |
| class + conf<0.5 | 836 | 102 | 0.340 | 0.637 | 0.444 | 0.172 | 0.479 |
| class + conf<0.7 | 836 | 102 | 0.153 | 0.931 | 0.264 | 0.714 | 0.637 |