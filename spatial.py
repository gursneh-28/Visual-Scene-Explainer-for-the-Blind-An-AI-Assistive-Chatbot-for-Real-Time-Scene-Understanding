def get_position(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    center_x  = (x1 + x2) / 2
    box_area   = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    size_ratio = box_area / frame_area

    # Horizontal
    if center_x < frame_width * 0.33:
        horizontal = "to your left"
    elif center_x > frame_width * 0.66:
        horizontal = "to your right"
    else:
        horizontal = "ahead of you"

    # Distance
    if size_ratio > 0.25:
        distance = "very close"
    elif size_ratio > 0.10:
        distance = "nearby"
    elif size_ratio > 0.03:
        distance = "a few steps away"
    else:
        distance = "far away"

    return {'horizontal': horizontal, 'distance': distance}


def describe_scene(detected, frame_width, frame_height):
    if not detected:
        return []

    descriptions = []
    for obj in detected:
        pos = get_position(obj['box'], frame_width, frame_height)
        # Short natural sentence: "Bottle ahead of you, very close"
        desc = f"{obj['label'].capitalize()} {pos['horizontal']}, {pos['distance']}"
        descriptions.append({
            'label':       obj['label'],
            'description': desc,
            'horizontal':  pos['horizontal'],
            'distance':    pos['distance']
        })

    return descriptions