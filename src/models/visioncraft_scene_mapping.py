from __future__ import annotations

VISIONCRAFT_SCENE_MAPPING_14 = {
    "bedroom": [
        "/b/bedroom",
        "/d/dorm_room",
        "/h/hotel_room",
        "/c/childs_room",
    ],
    "kitchen_dining": [
        "/k/kitchen",
        "/d/dining_room",
    ],
    "office_study": [
        "/o/office",
        "/o/office_cubicles",
        "/h/home_office",
        "/c/classroom",
        "/c/computer_room",
        "/l/lecture_room",
        "/k/kindergarden_classroom",
    ],
    "restaurant_cafe": [
        "/r/restaurant",
        "/c/coffee_shop",
        "/c/cafeteria",
        "/f/fastfood_restaurant",
        "/p/pizzeria",
        "/s/sushi_bar",
    ],
    "corridor_lobby": [
        "/c/corridor",
        "/l/lobby",
        "/e/elevator_lobby",
        "/e/entrance_hall",
    ],
    "street_downtown": [
        "/s/street",
        "/d/downtown",
    ],
    "transportation_hub_road": [
        "/h/highway",
        "/a/airport_terminal",
        "/a/airfield",
        "/b/bridge",
        "/r/railroad_track",
        "/s/subway_station/platform",
        "/t/train_station/platform",
    ],
    "residential_outdoor": [
        "/h/house",
        "/a/apartment_building/outdoor",
        "/r/residential_neighborhood",
        "/y/yard",
        "/p/porch",
        "/c/cottage",
    ],
    "forest_nature": [
        "/f/forest/broadleaf",
        "/r/rainforest",
        "/b/bamboo_forest",
    ],
    "mountain_valley": [
        "/m/mountain",
        "/m/mountain_snowy",
        "/v/valley",
        "/c/cliff",
        "/c/canyon",
    ],
    "waterfront": [
        "/b/beach",
        "/c/coast",
        "/l/lake/natural",
        "/r/river",
        "/w/waterfall",
    ],
    "open_field_landscape": [
        "/f/field/cultivated",
        "/f/field/wild",
        "/w/wheat_field",
        "/h/hayfield",
        "/s/sky",
        "/d/desert/sand",
        "/d/desert/vegetation",
        "/p/pasture",
        "/s/snowfield",
    ],
    "industrial_area": [
        "/i/industrial_area",
        "/c/construction_site",
        "/e/engine_room",
        "/a/assembly_line",
        "/a/auto_factory",
    ],
    "public_large_indoor": [
        "/a/auditorium",
        "/m/museum/indoor",
        "/n/natural_history_museum",
        "/s/science_museum",
        "/c/church/indoor",
        "/a/atrium/public",
    ],
}


VISIONCRAFT_SCENE_MAPPING_14_REFINED = {
    "bedroom": VISIONCRAFT_SCENE_MAPPING_14["bedroom"],
    "kitchen_dining": VISIONCRAFT_SCENE_MAPPING_14["kitchen_dining"],
    "office_study": VISIONCRAFT_SCENE_MAPPING_14["office_study"],
    "restaurant_cafe": VISIONCRAFT_SCENE_MAPPING_14["restaurant_cafe"],
    "corridor_lobby": [
        "/c/corridor",
        "/e/elevator_lobby",
    ],
    "street_downtown": VISIONCRAFT_SCENE_MAPPING_14["street_downtown"],
    "transportation_hub_road": VISIONCRAFT_SCENE_MAPPING_14["transportation_hub_road"],
    "residential_outdoor": VISIONCRAFT_SCENE_MAPPING_14["residential_outdoor"],
    "forest_nature": VISIONCRAFT_SCENE_MAPPING_14["forest_nature"],
    "mountain_valley": VISIONCRAFT_SCENE_MAPPING_14["mountain_valley"],
    "waterfront": [
        "/b/beach",
        "/c/coast",
        "/l/lake/natural",
    ],
    "open_field_landscape": VISIONCRAFT_SCENE_MAPPING_14["open_field_landscape"],
    "industrial_area": VISIONCRAFT_SCENE_MAPPING_14["industrial_area"],
    "public_large_indoor": [
        "/a/auditorium",
        "/m/museum/indoor",
        "/n/natural_history_museum",
        "/s/science_museum",
        "/c/church/indoor",
    ],
}


VISIONCRAFT_SCENE_MAPPING_10 = {
    "bedroom": VISIONCRAFT_SCENE_MAPPING_14["bedroom"],
    "food_space": (
        VISIONCRAFT_SCENE_MAPPING_14["kitchen_dining"]
        + VISIONCRAFT_SCENE_MAPPING_14["restaurant_cafe"]
    ),
    "office_study": VISIONCRAFT_SCENE_MAPPING_14["office_study"],
    "public_indoor": (
        VISIONCRAFT_SCENE_MAPPING_14["corridor_lobby"]
        + VISIONCRAFT_SCENE_MAPPING_14["public_large_indoor"]
    ),
    "street_downtown": VISIONCRAFT_SCENE_MAPPING_14["street_downtown"],
    "transportation_hub_road": VISIONCRAFT_SCENE_MAPPING_14["transportation_hub_road"],
    "residential_outdoor": VISIONCRAFT_SCENE_MAPPING_14["residential_outdoor"],
    "green_open_nature": (
        VISIONCRAFT_SCENE_MAPPING_14["forest_nature"]
        + VISIONCRAFT_SCENE_MAPPING_14["open_field_landscape"]
    ),
    "mountain_waterfront": (
        VISIONCRAFT_SCENE_MAPPING_14["mountain_valley"]
        + VISIONCRAFT_SCENE_MAPPING_14["waterfront"]
    ),
    "industrial_area": VISIONCRAFT_SCENE_MAPPING_14["industrial_area"],
}


MAPPING_PRESETS = {
    "14": VISIONCRAFT_SCENE_MAPPING_14,
    "14-refined": VISIONCRAFT_SCENE_MAPPING_14_REFINED,
    "10": VISIONCRAFT_SCENE_MAPPING_10,
    "16": {
        "bedroom": [
            "/b/bedroom",
            "/d/dorm_room",
            "/h/hotel_room",
        ],
        "kitchen": [
            "/k/kitchen",
        ],
        "office_workspace": [
            "/o/office",
            "/o/office_cubicles",
            "/h/home_office",
        ],
        "classroom_computer_room": [
            "/c/classroom",
            "/c/computer_room",
            "/k/kindergarden_classroom",
        ],
        "restaurant_cafe": [
            "/r/restaurant",
            "/c/coffee_shop",
            "/c/cafeteria",
            "/f/fastfood_restaurant",
            "/p/pizzeria",
            "/s/sushi_bar",
        ],
        "corridor_lobby": [
            "/c/corridor",
            "/l/lobby",
            "/e/elevator_lobby",
            "/e/entrance_hall",
        ],
        "museum_church_auditorium": [
            "/m/museum/indoor",
            "/n/natural_history_museum",
            "/s/science_museum",
            "/c/church/indoor",
            "/a/auditorium",
        ],
        "street_downtown": [
            "/s/street",
            "/d/downtown",
        ],
        "transportation_infra": [
            "/h/highway",
            "/b/bridge",
            "/r/railroad_track",
            "/s/subway_station/platform",
            "/t/train_station/platform",
        ],
        "airport_airfield": [
            "/a/airport_terminal",
            "/a/airfield",
        ],
        "residential_outdoor": [
            "/h/house",
            "/a/apartment_building/outdoor",
            "/r/residential_neighborhood",
            "/y/yard",
            "/p/porch",
            "/c/cottage",
        ],
        "industrial_area": [
            "/i/industrial_area",
            "/c/construction_site",
            "/e/engine_room",
            "/a/assembly_line",
            "/a/auto_factory",
        ],
        "forest": [
            "/f/forest/broadleaf",
            "/r/rainforest",
            "/b/bamboo_forest",
        ],
        "mountain_valley": [
            "/m/mountain",
            "/m/mountain_snowy",
            "/v/valley",
            "/c/cliff",
            "/c/canyon",
        ],
        "waterfront": [
            "/b/beach",
            "/c/coast",
            "/l/lake/natural",
            "/r/river",
            "/w/waterfall",
        ],
        "open_landscape": [
            "/f/field/cultivated",
            "/f/field/wild",
            "/w/wheat_field",
            "/h/hayfield",
            "/s/sky",
            "/d/desert/sand",
            "/d/desert/vegetation",
            "/p/pasture",
            "/s/snowfield",
        ],
    },
}

VISIONCRAFT_SCENE_MAPPING = VISIONCRAFT_SCENE_MAPPING_14


def get_mapping(mapping_version: str = "14") -> dict[str, list[str]]:
    if mapping_version not in MAPPING_PRESETS:
        raise ValueError(f"Unknown mapping version: {mapping_version}")
    return MAPPING_PRESETS[mapping_version]


def build_reverse_mapping(mapping_version: str = "14") -> dict[str, str]:
    mapping = get_mapping(mapping_version)
    reverse_mapping: dict[str, str] = {}
    for target_class, source_classes in mapping.items():
        for source_class in source_classes:
            reverse_mapping[source_class] = target_class
    return reverse_mapping
