| stage | model | FPS |
|---|---|---|
| segmentation | deeplab-r50 | 19.4 |
| segmentation | segformer-b0 | 22.3 |
| segmentation | segformer-b2 | 14.8 |
| detection | yolo11m_rail | 34.8 |
| detection | yolo11s_rail | 47.3 |
| full pipeline | default config, segmentation every 1 frame(s) | 7.4 |
| full pipeline | default config, segmentation every 3 frame(s) | 15.0 |

GPU: NVIDIA GeForce RTX 5060 Laptop GPU; frames 960x540; mean of 100 frames after 10 warm-up frames.