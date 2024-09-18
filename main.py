import os
import argparse
from platform import processor
from modules.trainModel import trainModel
from modules.prepareDataset import prepareDataset
from transformers import Trainer, TrainingArguments
import torch
from torchvision import transforms
import numpy as np
import datasets as datasets

def transform(example_batch):
   desired_size = (224, 224)
   resized_images = [transforms.Resize(desired_size)(x.convert("RGB")) for x in example_batch['image']]
   inputs = processor(resized_images, return_tensors='pt')
   inputs['label'] = example_batch['label']
   return inputs

def prepare_data(custom_dataset_dir):
   ds = prepareDataset.load_custom_dataset(custom_dataset_dir)
   prepared_ds = ds.with_transform(transform)
   return ds, prepared_ds

def verify_data(custom_dataset_dir):
    ds = prepareDataset.load_custom_dataset(custom_dataset_dir)
    print("Resolving data files:", end=" ")
    # Simulate progress for the sake of example
    for i in range(2088):
        if i % 100 == 0:
            print(f"{i/2088:.0%}", end=' ')
    print()
    # Simulate download time
    print("Downloading data:", end=" ")
    for i in range(2088):
        if i % 100 == 0:
            print(f"{i/2088:.0%}", end=' ')
    print()
    # Simulate dataset split generation time
    print("Generating train split:", end=" ")
    for i in range(2088):
        if i % 100 == 0:
            print(f"{i/2088:.0%}", end=' ')
    print()
    # Print dataset structure
    print("DatasetDict({ train:", ds['train'].num_rows, "test:", ds['test'].num_rows, "valid:", ds['valid'].num_rows, "})")

def main():
   script_dir = os.path.dirname(__file__)
   custom_dataset_dir = os.path.join(script_dir, 'training/custom_dataset/')
   output_dir = os.path.join(script_dir, 'training/output/')

   parser = argparse.ArgumentParser(description="Train a ViT model on a custom dataset.")
   parser.add_argument("-prepareData", action='store_true', help="Prepare the custom dataset")
   parser.add_argument("-train", action='store_true', help="Train the model")
   parser.add_argument("-verifyData", action='store_true', help="Verify the data before training")
   args = parser.parse_args()

   if args.verifyData:
       verify_data(custom_dataset_dir)
       return  # Exit after verification to prevent model training or preparation   

   if args.prepareData:
       ds, prepared_ds = prepare_data(custom_dataset_dir)
       labels = ds['train'].features['label']
       processor, model = trainModel.load_model('google/vit-base-patch16-224-in21k', labels)
       
       # Define training arguments
       training_args = TrainingArguments(
           output_dir=output_dir,
           per_device_train_batch_size=16,
           evaluation_strategy="epoch",
           save_strategy="epoch",
           fp16=True,
           num_train_epochs=20,
           logging_steps=500,
           learning_rate=2e-4,
           save_total_limit=1,
           remove_unused_columns=False,
           push_to_hub=False,
           report_to='tensorboard',
           load_best_model_at_end=True,
       )
       
       # Define collate function and metric function
       def collate_fn(batch):
           return {
               'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
               'labels': torch.tensor([x['label'] for x in batch])
           }
       
       metric = datasets.load_metric("accuracy")
       
       def compute_metrics(p):
           return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
       
       # Define Trainer and train the model
       trainer = Trainer(
           model=model,
           args=training_args,
           data_collator=collate_fn,
           compute_metrics=compute_metrics,
           train_dataset=prepared_ds["train"],
           eval_dataset=prepared_ds["valid"],  # Use valid dataset for evaluation during training
           tokenizer=processor,
       )
       
       # Train the model   
       train_results = trainer.train()
       trainer.save_model(output_dir)  # Save the best model
       trainer.log_metrics("train", train_results.metrics)
       trainer.save_metrics("train", train_results.metrics)
       trainer.save_state()

   if args.evaluateModel:
       ds, prepared_ds = prepare_data(custom_dataset_dir)
       labels = ds['train'].features['label']
       processor, model = trainModel.load_model('google/vit-base-patch16-224-in21k', labels)
       
       # Define training arguments (this part is just for demonstration and won't be used if -train is specified)
       training_args = TrainingArguments(
           output_dir=output_dir,
           per_device_train_batch_size=16,
           evaluation_strategy="epoch",
           save_strategy="epoch",
           fp16=True,
           num_train_epochs=20,
           logging_steps=500,
           learning_rate=2e-4,
           save_total_limit=1,
           remove_unused_columns=False,
           push_to_hub=False,
           report_to='tensorboard',
           load_best_model_at_end=True,
       )
       
       # Define collate function and metric function (these won't be used in evaluation mode)
       def collate_fn(batch):
           return {
               'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
               'labels': torch.tensor([x['label'] for x in batch])
           }
       
       metric = datasets.load_metric("accuracy")
       
       def compute_metrics(p):
           return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
       
       # Evaluate the model on the test dataset
       trainer = Trainer(
           model=model,
           args=training_args,
           data_collator=collate_fn,
           compute_metrics=compute_metrics,
           train_dataset=prepared_ds["train"],
           eval_dataset=prepared_ds["test"],  # Use test dataset for evaluation
           tokenizer=processor,
       )
       
       eval_results = trainer.evaluate()
       print(eval_results)

if __name__ == "__main__":
   main()