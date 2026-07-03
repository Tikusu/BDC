import json

RULES = {
    "Recyclable Objects": [
        "bottle", "can", "carton", "box", "crate", "paper", "newspaper", "book",
        "jar", "barrel", "packet", "bag", "cardboard", "wine bottle", "beer bottle",
        "pop bottle", "milk can", "tin", "plastic bag", "envelope", "pill bottle",
    ],
    "Electronic Objects": [
        "laptop", "computer", "keyboard", "mouse", "monitor", "screen", "phone",
        "cellular", "television", "tv", "remote", "printer", "camera", "modem",
        "router", "speaker", "microphone", "hard disc", "disk drive", "joystick",
        "calculator", "digital", "iPod", "CD player", "radio", "hand-held",
        "electric", "washer", "dishwasher", "microwave", "toaster", "space heater",
        "hair dryer", "vacuum", "power drill", "cassette player", "projector",
        "oscilloscope", "dial telephone", "desktop computer", "notebook", "CPU",
    ],
    "Organic Objects": [
        "banana", "orange", "lemon", "fig", "pineapple", "strawberry", "apple",
        "corn", "mushroom", "vegetable", "fruit", "broccoli", "cauliflower",
        "cucumber", "artichoke", "bell pepper", "pomegranate", "zucchini",
        "squash", "cabbage", "acorn", "leaf", "flower", "daisy", "rose",
        "banana", "custard apple", "buckeye", "hip", "potato", "tomato"
    ],
}

def main():
    classes = [l.strip() for l in open("imagenet_classes.txt") if l.strip()]
    mapping = {}
    for idx, name in enumerate(classes):
        lname = name.lower()
        assigned = None
        for category, keywords in RULES.items():
            if any(kw in lname for kw in keywords):
                assigned = category
                break
        mapping[str(idx)] = {"class_name": name, "category": assigned}

    with open("category_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    counts = {}
    for v in mapping.values():
        counts[v["category"]] = counts.get(v["category"], 0) + 1
    print(counts)

if __name__ == "__main__":
    main()