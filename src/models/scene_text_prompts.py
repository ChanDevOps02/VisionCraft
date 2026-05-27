from __future__ import annotations


SCENE_CLASS_NAMES = [
    "bedroom",
    "corridor_lobby",
    "forest_nature",
    "industrial_area",
    "kitchen_dining",
    "mountain_valley",
    "office_study",
    "open_field_landscape",
    "public_large_indoor",
    "residential_outdoor",
    "restaurant_cafe",
    "street_downtown",
    "transportation_hub_road",
    "waterfront",
]


SCENE_TEXT_KEYWORDS_V1 = {
    "bedroom": "bed, pillow, blanket, wall, floor, lamp, window, cabinet, indoor room",
    "corridor_lobby": "wall, floor, ceiling, door, column, window, indoor corridor, lobby, hallway",
    "forest_nature": "tree, plant, grass, earth, path, sky, natural forest, vegetation",
    "industrial_area": "building, road, truck, car, concrete, metal, wall, warehouse, industrial outdoor area",
    "kitchen_dining": "table, chair, cabinet, counter, sink, refrigerator, wall, floor, indoor kitchen dining room",
    "mountain_valley": "mountain, rock, earth, sky, tree, grass, valley, natural landscape",
    "office_study": "desk, chair, screen, computer, cabinet, table, wall, floor, indoor office study room",
    "open_field_landscape": "grass, earth, sky, field, plant, open outdoor landscape, plain",
    "public_large_indoor": "floor, wall, ceiling, column, people, sign, wide indoor hall, public interior",
    "residential_outdoor": "house, building, road, sidewalk, tree, fence, residential outdoor neighborhood",
    "restaurant_cafe": "table, chair, counter, bottle, cup, wall, floor, indoor restaurant cafe",
    "street_downtown": "building, road, sidewalk, car, signboard, person, urban street, downtown",
    "transportation_hub_road": "road, car, bus, truck, sign, bridge, station, transportation hub, roadway",
    "waterfront": "sky, water, sea, rock, sand, coast, shoreline, waves, waterfront outdoor scene",
}


SCENE_TEXT_PROMPTS_V1 = {
    "bedroom": "a bedroom scene with bed, pillow, blanket, wall, floor, lamp, and indoor resting space",
    "corridor_lobby": "an indoor corridor or lobby scene with walls, floor, ceiling, doors, and hallway-like space",
    "forest_nature": "a natural forest scene with trees, plants, grass, earth, and outdoor vegetation",
    "industrial_area": "an industrial outdoor area with buildings, roads, concrete, metal structures, trucks, and warehouses",
    "kitchen_dining": "an indoor kitchen or dining scene with tables, chairs, cabinets, counters, sink, and appliances",
    "mountain_valley": "a mountain or valley landscape with rocks, earth, sky, grass, and natural terrain",
    "office_study": "an indoor office or study scene with desk, chair, screen, computer, cabinet, and workspace",
    "open_field_landscape": "an open outdoor field landscape with grass, earth, sky, and wide open space",
    "public_large_indoor": "a large public indoor scene with floor, walls, ceiling, columns, people, signs, and open interior space",
    "residential_outdoor": "a residential outdoor neighborhood scene with houses, buildings, roads, sidewalks, trees, and fences",
    "restaurant_cafe": "an indoor restaurant or cafe scene with tables, chairs, counter space, cups, bottles, and dining atmosphere",
    "street_downtown": "a downtown street scene with buildings, roads, sidewalks, cars, signboards, and people",
    "transportation_hub_road": "a transportation hub or roadway scene with roads, vehicles, signs, stations, and transit-related structures",
    "waterfront": "a waterfront outdoor scene with sky, sea, water, rocks, shoreline, waves, and coastal landscape",
}


def get_scene_text_prompts(style: str = "sentence_v1") -> dict[str, str]:
    if style == "sentence_v1":
        return dict(SCENE_TEXT_PROMPTS_V1)
    if style == "keyword_v1":
        return dict(SCENE_TEXT_KEYWORDS_V1)
    raise ValueError(f"Unsupported prompt style: {style}")

