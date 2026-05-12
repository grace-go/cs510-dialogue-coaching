from load_data import load_json, save_json
from tagger import tag_user_input


INPUT_PATH = "data/inputs/test_inputs.json"
TAG_PATH = "data/corpus/tag_definitions.json"
OUTPUT_PATH = "data/inputs/tagged_test_inputs.json"


def main():
    tag_definitions = load_json(TAG_PATH)
    inputs = load_json(INPUT_PATH)

    tagged_inputs = []

    for item in inputs:
        input_id = item["id"]
        scenario = item["scenario"]
        transcript = item["transcript"]

        print(f"Tagging {input_id}...")

        predicted_tags = tag_user_input(
            transcript=transcript,
            scenario=scenario,
            tag_definitions=tag_definitions
        )

        tagged_item = {
            **item,
            "predicted_tags": predicted_tags
        }

        tagged_inputs.append(tagged_item)

        # Save after every input so progress is not lost
        save_json(tagged_inputs, OUTPUT_PATH)

    print(f"Saved tagged inputs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()