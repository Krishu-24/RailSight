| stage | model | FPS |
|---|---|---|
| segmentation | deeplab-r50 | 29.8 |
| segmentation | segformer-b0 | 58.7 |
| segmentation | segformer-b2 | 25.4 |
| detection | rtdetr_l_rail @640px | 36.1 |
| detection | yolo11m_rail @640px | 69.8 |
| detection | yolo11m_rail_v2 @640px | 72.8 |
| detection | yolo11m_rail_v2_960 @960px | 76.5 |
| detection | yolo11s_rail @640px | 102.3 |
| full pipeline | default config, segmentation every 1 frame(s) | 15.0 |
| full pipeline | default config, segmentation every 3 frame(s) | 28.0 |

GPU: NVIDIA GeForce RTX 5060 Laptop GPU; frames 960x540; mean of 100 frames after 10 warm-up frames.