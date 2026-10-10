| stage | model | FPS |
|---|---|---|
| segmentation | deeplab-r50 | 27.7 |
| segmentation | segformer-b0 | 53.8 |
| segmentation | segformer-b2 | 21.0 |
| detection | yolo11m_rail | 45.6 |
| detection | yolo11s_rail | 57.0 |
| full pipeline | default config, segmentation every 1 frame(s) | 9.8 |
| full pipeline | default config, segmentation every 3 frame(s) | 17.7 |

GPU: NVIDIA GeForce RTX 5060 Laptop GPU; frames 960x540; mean of 100 frames after 10 warm-up frames.