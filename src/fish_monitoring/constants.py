"""Project-wide constants (class names, default splits).

``CLASS_NAMES`` mirrors the ``names`` list of the released WIO-ReefFish
``data.yaml`` — 24 fish families, alphabetically ordered. It is only used as a
fallback default; every script that receives a ``data.yaml`` reads the class
names from there instead.
"""

CLASS_NAMES = [
    "Acanthuridae", "Anthiadidae", "Aulostomidae", "Balistidae", "Caesionidae",
    "Chaetodontidae", "Epinephelidae", "Haemulidae", "Holocentridae", "Kyphosidae",
    "Labridae", "Lethrinidae", "Lutjanidae", "Microdesmidae", "Monacanthidae",
    "Mullidae", "Pempheridae", "Pomacanthidae", "Pomacentridae", "Priacanthidae",
    "Scarinae", "Siganidae", "Tetraodontidae", "Zanclidae",
]

DEFAULT_SPLITS = ("train", "valid", "test")
