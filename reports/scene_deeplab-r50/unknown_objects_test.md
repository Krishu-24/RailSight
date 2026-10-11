| variant | frames | frames_with_obstacle | precision | recall | F1 | false_alarm_rate | object_recall |
|---|---|---|---|---|---|---|---|
| class only | 845 | 103 | 0.575 | 0.485 | 0.526 | 0.050 | 0.399 |
| class + conf<0.3 | 845 | 103 | 0.568 | 0.485 | 0.524 | 0.051 | 0.399 |
| class + conf<0.5 | 845 | 103 | 0.309 | 0.660 | 0.421 | 0.205 | 0.453 |
| class + conf<0.7 | 845 | 103 | 0.142 | 0.942 | 0.247 | 0.790 | 0.622 |