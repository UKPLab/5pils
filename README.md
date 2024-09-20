# Image, Tell me your story! Predicting the original meta-context of visual misinformation

[![Arxiv](https://img.shields.io/badge/Arxiv-2408.09939-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2408.09939)
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This repository contains the 5Pils dataset, introduced in the paper: ["Image, Tell me your story!" Predicting the original meta-context of visual misinformation](https://arxiv.org/abs/2408.09939). It also contains the code to run experiments with the baseline introduced in the same paper. The code is released under an **Apache 2.0** license, while the dataset is realeased under a **CC-BY-SA-4.0** license.

Contact person: [Jonathan Tonglet](mailto:jonathan.tonglet@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

The dataset is also available on [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4317).

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions. 

## News 📢

- Our paper is accepted to EMNLP 2024 Main Conference! See you in Miami 🏖️

## Abstract 
> To assist human fact-checkers, researchers have developed automated approaches for visual misinformation detection. These methods assign veracity scores by identifying inconsistencies between the image and its caption, or by detecting forgeries in the image. However, they neglect a crucial point of the human factchecking process: identifying the original metacontext of the image. By explaining what is *actually true* about the image, fact-checkers can better detect misinformation, focus their efforts on check-worthy visual content, engage in counter-messaging before misinformation spreads widely, and make their explanation more convincing. Here, we fill this gap by introducing the task of automated image contextualization. We create 5Pils, a dataset of 1,676 fact-checked images with questionanswer pairs about their original meta-context. Annotations are based on the 5 Pillars factchecking framework. We implement a first baseline that grounds the image in its original meta-context using the content of the image and textual evidence retrieved from the open web. Our experiments show promising results while highlighting several open challenges in retrieval and reasoning.

<p align="center">
  <img width="70%" src="assets/introducory_example.png" alt="header" />
</p>



## 5Pils dataset

The 5Pils dataset consists of 1,676 fact-checked images annotated with question-answer pairs based on the 5 Pillars framework for image contextualization. The dataset annotations are contained in the train.json, val.json, and test.json files of the dataset folder. 
More information about the dataset structure can be found in the README file of the dataset folder.

### ❕**Content warning** ❕
> 5Pils contains examples of real-world misinformation.  Due to the real-world nature of the data, events covered include wars and conflicts. As a result, some images contain graphic, violent content. When collecting the data, we decided not to filter out images with violent content to cover the actual distribution of images that our target users, professional fact-checkers, would want to provide as input. Given
the graphic nature of some images, we do not release them directly. Instead, we do publicly release the URLs of the FC articles, as well as the script that allows to collect and process the images.

### 5Pils example

````json
    {
        "URL": "https://factly.in/2013-evacuation-image-from-typhoon-hit-philippines-is-passed-off-as-iaf-airlifting-800-people-from-kabul",
        "image path": "dataset/processed_img/2013-evacuation-image-from-typhoon-hit-philippines-is-passed-off-as-iaf-airlifting-800-people-from-kabul.png",
        "org": "factly",
        "publication date": "2021-08-17T12:00:55+00:00",
        "claim": "The image shows 800 people airlifted by the Indian Air Force from Kabul.",
        "was the photo used before?": "yes",
        "source": "US Airforce's official website",
        "date": "2013",
        "date numeric label": "['2013-01-14T00:00:00+00:00']",
        "location": "Tacloban city, Philippines",
        "motivation": "To document the US Airforce\u2019s operation Damayan evacuation in 2013",
        "type of image": "out-of-context",
        "verification strategy": "['reverse image search']",
        "verification tool": "['Other/Unspecified']",
        "claimed location": "Kabul, Afghanistan",
        "claimed date": "2021"
    }

````

## Getting started

### Environment

Follow these instructions to recreate thy python environment used for all our experiments. All experiments with Llava and Llama2 ran on A100 GPUs.

```
$ conda create --name 5Pils python=3.9
$ conda activate 5Pils
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_lg
```

### Google Vision API and Azure OpenAI service

- Our baseline model relies on evidence retrieved with reverse image search using the [Google Vision API](https://cloud.google.com/vision/docs/detecting-web). All URLs for the text evidence that we used in our experiments are provided in this repo. However, should you want to collect your own evidence, you will need to create a Google Cloud account and create an API key, or use a different reverse image search service.

- If you want to use GPT4(-Vision) for answer generation, you will need an Azure account with access to the [Azure OpenAI service](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview).


## Usage - dataset



To use the dataset, you need to collect the images based on the 1,676 FC article urls provided in dataset/url/article_urls.txt:

```
$ python scripts/build_dataset_from_url.py
```

## Usage - evaluation

Evaluate the performance of a model on 5Pils, for a specific pillar. In this example, we conduct the evaluation for Date.

```
$ python scripts/evaluate_answer_generation.py --results_file output/results_date.json --task date
```
 
Evaluation of Location requires 🌍 [GeoNames](https://www.geonames.org/). You will need to create a (free) account and provide your account name as input.

## Usage - baseline 

<p align="center">
  <img width="75%" src="assets/baseline.png" alt="header" />
</p>

### Collecting the evidence

Collect the text evidence based on their URLS:

```
$ python scripts/collect_RIS_evidence.py --collect_google 0 --evidence_urls dataset/retrieval_results/evidence_urls.json 
```

Instead of using our evidence set, you can also collect your own by setting *collect_google* to 1. This will require to provide a Google Vision API key. 


### Predict and collect the original of manipulated images

Step 1 (Optional): Classify images as manipulated or not after fine-tuning a ViT model on the train set. This step is optional as the predictions for the test set are already provided in *dataset/manipulation_detection_test.json*

```
$ python scripts/get_manipulated_images.py
```

Step 2: Replace manipulated images by the earliest corresponding reverse image search result:

```
$ python scripts/get_originals_of_manipulated.py --download_image 1
```

###  Compute embeddings 

Compute embeddings for evidence ranking and few-shot demonstration selection:

```
$ python scripts/get_embeddings.py
```

### Generate answers

Generate answers for a specific pillar using a LLM or Multimodal LLM. In this example, we generate answers for Date with multimodal zero-shot Llava:

```
$ python scripts/get_5pillars_answers.py --results_file output/results_date.json --task date --modality multimodal --n_shots 0 --model llava
```

### Evaluate the evidence ranking

Evaluate the quality of the evidence ranking:

```
$ python scripts/evaluation_ranking.py
```

## Citation

If you use the 5Pils dataset or this code in your work, please cite our paper as follows:

```bibtex 
@article{tonglet2024imagetellstorypredicting,
  title={"Image, Tell me your story!" Predicting the original meta-context of visual misinformation},
  author={Tonglet, Jonathan and Moens, Marie-Francine and Gurevych, Iryna},
  journal={arXiv preprint arXiv:2408.09939},
  year={2024},
  doi={10.48550/arXiv.2408.09939},
  url={https://www.arxiv.org/abs/2408.09939}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.




