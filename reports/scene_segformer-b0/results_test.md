| variant | frames | frames_with_obstacle | precision | recall | F1 | false_alarm_rate | object_recall |
|---|---|---|---|---|---|---|---|
| class only | 845 | 103 | 0.597 | 0.359 | 0.448 | 0.034 | 0.236 |
| class + conf<0.3 | 845 | 103 | 0.539 | 0.398 | 0.458 | 0.047 | 0.250 |
| class + conf<0.5 | 845 | 103 | 0.210 | 0.631 | 0.315 | 0.330 | 0.331 |
| class + conf<0.7 | 845 | 103 | 0.126 | 0.951 | 0.222 | 0.916 | 0.588 |