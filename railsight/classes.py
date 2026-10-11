"""RailSem19 class ids and the groups the pipeline uses. Plain data: no deep-learning imports,
so geometry and logic modules (and their tests) stay lightweight."""

RS19_CLASSES = [
    "road", "sidewalk", "construction", "tram-track", "fence", "pole", "traffic-light",
    "traffic-sign", "vegetation", "terrain", "sky", "human", "rail-track", "car", "truck",
    "trackbed", "on-rails", "rail-raised", "rail-embedded",
]  # fmt: skip
IGNORE_INDEX = 255
NUM_CLASSES = len(RS19_CLASSES)
CLS = {n: i for i, n in enumerate(RS19_CLASSES)}

# Pixels that make up a drivable track (used to find the ego path)
TRACK_IDS = [CLS["rail-track"], CLS["tram-track"], CLS["rail-raised"], CLS["rail-embedded"]]
# The rails themselves (used to measure the gauge between the two rails of the ego track)
RAIL_IDS = [CLS["rail-raised"], CLS["rail-embedded"]]
# Surfaces that are normal to see on or between the rails; anything else there is foreign
GROUND_IDS = TRACK_IDS + [CLS[c] for c in ("trackbed", "terrain", "vegetation", "road", "sidewalk")]
