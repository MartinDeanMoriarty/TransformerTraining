from transformers import AutoImageProcessor, ViTForImageClassification

def load_model(model_name_or_path, labels):
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True
    )
    return processor, model