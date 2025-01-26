import numpy as np
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, pipeline
import torch
from PIL import Image
from datasets import load_metric, Dataset
import argparse
import os
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def transform(example_batch):
    '''
    Take a list of PIL images and turn them to pixel values
    '''
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['labels']
    return inputs


def collate_fn(batch):
    label_map = {'manipulated': 1, 'non-manipulated': 0}

    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([label_map[x['labels']] for x in batch])
    }


def train_manipulation_detector(prepared_train_dataset,
                                prepared_val_dataset,
                                model_name_or_path,
                                save_folder,
                                epochs,
                                learning_rate):
    '''
    Fine-tune a Vision Transformer for classifying images as manipulated or non-manipulated.
    '''
    labels = ["non-manipulated", "manipulated"]
    model = ViTForImageClassification.from_pretrained(
                                                    model_name_or_path,
                                                    num_labels=len(labels),
                                                    id2label={str(i): c for i, c in enumerate(labels)},
                                                    label2id={c: str(i) for i, c in enumerate(labels)}
                                                    )
    training_args = TrainingArguments(
    output_dir=save_folder,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=epochs,
    fp16=True,
    save_steps=500,
    eval_steps=50,
    logging_steps=500,
    learning_rate=learning_rate,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_train_dataset,
        eval_dataset=prepared_val_dataset,
        tokenizer=processor,
    )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Classify images as manipulated or not with a ViT model.')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k',
                        help='The ViT version to use') 
    parser.add_argument('--train',type=int, default=0,
                        help='If True, train a ViT model. If False, load an existing model in the location specified by model_folder.')
    parser.add_argument('--model_folder', type=str, default='baseline/vit-manipulation',
                        help='Folder to save and load the trained model') 
    parser.add_argument('--json_path', type=str, default='dataset/manipulation_detection_test.json',
                        help='File to save the model predictions') 
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs') 
    parser.add_argument('--learning_rate', type=int, default=2e-4,
                        help='Learning rate for training') 
    

    args = parser.parse_args()  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #Load datasets
    train = load_json('dataset/train.json')
    val = load_json('dataset/val.json')
    test = load_json('dataset/test.json')

    #Update image path
    train_input = [im['image_path'] for im in train]
    val_input = [im['image_path'] for im in val]
    test_input = [im['image_path'] for im in test]
    #change type of image label
    train_target = ['manipulated' if im['type_of_image']=='manipulated' else 'non-manipulated' for im in train]
    val_target = ['manipulated' if im['type_of_image']=='manipulated' else 'non-manipulated' for im in val]
    test_target = ['manipulated' if im['type_of_image']=='manipulated' else 'non-manipulated' for im in test]
    #Convert to HF dataset
    train_dataset = Dataset.from_dict({'image_path': train_input,'labels': train_target})
    val_dataset = Dataset.from_dict({'image_path': val_input,'labels': val_target})
    test_dataset = Dataset.from_dict({'image_path': test_input,'labels': test_target})


    # Preprocessing
    model_name_or_path = args.model_name
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)
    prepared_train_dataset = train_dataset.with_transform(transform)
    prepared_val_dataset = val_dataset.with_transform(transform)
    prepared_test_dataset = test_dataset.with_transform(transform)

    #Load model
    if args.train:
        #Load metrics
        metric = load_metric("accuracy")
        train_manipulation_detector(prepared_train_dataset,prepared_val_dataset,
                                    args.model_name, args.model_folder,
                                    args.epochs, args.learning_rate)
        
    ###############
    #  INFERENCE  #
    ###############

    # #Load existing model
    model = ViTForImageClassification.from_pretrained(args.model_folder)
    pipe = pipeline('image-classification',model=args.model_folder)
    #Load test set and make predictions
    test_loader = DataLoader(prepared_test_dataset, batch_size=32)
    test_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values']
            outputs = model(pixel_values)
            preds = torch.argmax(outputs.logits, dim=1)
            test_predictions.extend(preds.tolist())
    test_predictions = ['non-manipulated' if p==0 else 'manipulated' for p in test_predictions]
    results = [{'image_path':test[im]['image_path'], 'manipulation_detection':test_predictions[im]} for im in range(len(test))]
    save_result(results,args.json_path)
