| variant | frames | frames_with_obstacle | precision | recall | F1 | false_alarm_rate | object_recall |
|---|---|---|---|---|---|---|---|
| class only | 836 | 102 | 0.595 | 0.490 | 0.538 | 0.046 | 0.404 |
| class + conf<0.3 | 836 | 102 | 0.588 | 0.490 | 0.535 | 0.048 | 0.404 |
| class + conf<0.5 | 836 | 102 | 0.312 | 0.657 | 0.423 | 0.202 | 0.459 |
| class + conf<0.7 | 836 | 102 | 0.142 | 0.941 | 0.247 | 0.787 | 0.623 |