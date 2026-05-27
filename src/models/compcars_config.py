from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompCarsHierarchyConfig:
    min_make_samples: int = 200
    min_model_samples: int = 80
    min_models_per_make: int = 3
    train_ratio: float = 0.8
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


# Initial non-China allowlist for a practical first-stage prototype.
# The final usable set should still be filtered by sample count after parsing
# the downloaded CompCars metadata.
NON_CHINA_BRAND_ALLOWLIST = {
    "acura",
    "alfa romeo",
    "aston martin",
    "audi",
    "bentley",
    "bmw",
    "buick",
    "cadillac",
    "chevrolet",
    "chrysler",
    "citroen",
    "dodge",
    "ferrari",
    "fiat",
    "ford",
    "gmc",
    "honda",
    "hyundai",
    "infiniti",
    "jaguar",
    "jeep",
    "kia",
    "lamborghini",
    "land rover",
    "lexus",
    "lincoln",
    "maserati",
    "mazda",
    "mercedes benz",
    "mercedes-benz",
    "mini",
    "mitsubishi",
    "nissan",
    "opel",
    "peugeot",
    "porsche",
    "renault",
    "rolls royce",
    "rolls-royce",
    "saab",
    "seat",
    "skoda",
    "subaru",
    "suzuki",
    "tesla",
    "toyota",
    "volkswagen",
    "volvo",
}


BRAND_NAME_ALIASES = {
    "benz": "mercedes-benz",
    "mercedes benz": "mercedes-benz",
    "rolls royce": "rolls-royce",
    "vw": "volkswagen",
}


def normalize_brand_name(name: str) -> str:
    normalized = " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())
    return BRAND_NAME_ALIASES.get(normalized, normalized)
