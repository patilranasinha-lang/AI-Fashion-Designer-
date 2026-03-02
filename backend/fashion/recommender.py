import json

def recommend_outfits(body_shape, skin_tone, occasion):
    """
    Recommends outfits based on the user's features and the specified occasion.

    Args:
        body_shape: The user's body shape.
        skin_tone: The user's skin tone.
        occasion: The occasion for the outfit.

    Returns:
        A list of recommended outfits.
    """
    with open("fashion/knowledge_base.json", "r") as f:
        knowledge_base = json.load(f)

    recommendations = []
    for item in knowledge_base:
        if body_shape in item["body_shapes"] and \
           skin_tone in item["skin_tones"] and \
           occasion in item["occasions"]:
            recommendations.append(item)

    return recommendations
