import json
import csv
from collections import defaultdict

def coco_to_csv(coco_json_path, output_csv_path):

    # read COCO JSON
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # mapping: category_id -> category_name
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # mapping: image_id -> image_name
    image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

    # mapping: image_id -> set of labels (multiple labels possible)
    image_labels = defaultdict(set)

    # collect labels for each image from annotations
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        label_name = cat_id_to_name[category_id]
        image_labels[image_id].add(label_name)

    # write the results to CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "labels"])  # header

        for image_id, labels in image_labels.items():
            image_name = image_id_to_name[image_id]
            labels_str = "|".join(sorted(labels))
            writer.writerow([image_name, labels_str])

    print("CSV file created:", output_csv_path)
# Example usage
coco_to_csv(
    r"E:\Data Dental X_Ray_Panoramic\Dental segmentation_X-ray panoramic\valid_annotations.coco.json",
    r"E:\Data Dental X_Ray_Panoramic\Dental segmentation_X-ray panoramic\Data Dental X_Ray_Panoramic\valid\valid_labels.csv"
)