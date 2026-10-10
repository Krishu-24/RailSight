| variant | frames | frames_with_obstacle | precision | recall | F1 | false_alarm_rate | object_recall |
|---|---|---|---|---|---|---|---|
| class only | 836 | 102 | 0.617 | 0.363 | 0.457 | 0.031 | 0.233 |
| class + conf<0.3 | 836 | 102 | 0.586 | 0.402 | 0.477 | 0.040 | 0.247 |
| class + conf<0.5 | 836 | 102 | 0.212 | 0.627 | 0.317 | 0.324 | 0.329 |
| class + conf<0.7 | 836 | 102 | 0.126 | 0.951 | 0.223 | 0.916 | 0.582 |