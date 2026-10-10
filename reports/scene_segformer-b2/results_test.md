| variant | frames | frames_with_obstacle | precision | recall | F1 | false_alarm_rate | object_recall |
|---|---|---|---|---|---|---|---|
| class only | 845 | 103 | 0.576 | 0.476 | 0.521 | 0.049 | 0.446 |
| class + conf<0.3 | 845 | 103 | 0.573 | 0.495 | 0.531 | 0.051 | 0.446 |
| class + conf<0.5 | 845 | 103 | 0.330 | 0.631 | 0.433 | 0.178 | 0.480 |
| class + conf<0.7 | 845 | 103 | 0.153 | 0.932 | 0.263 | 0.716 | 0.635 |