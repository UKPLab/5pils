# The 5Pils dataset 📸

The 5Pils dataset contains the annotated meta-context of 1,676 fact-checked images. The dataset is split in 3 json files containing  the train, val, and test sets.
5Pils is made available under a **CC-BY-SA-4.0** license. Instructions to download the images are provided [here](https://github.com/UKPLab/5pils/blob/main/README.md#usage---dataset). 

```
$ conda create --name 5Pils python=3.9
$ conda activate 5Pils
$ pip install -r requirements.txt
$ python -m spacy download en_core_web_lg
$ python scripts/build_dataset_from_url.py
```

The dataset is also available on [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4317).

Each record contains the following items: 

**Main items**

- `URL`: URL leading to the fact-checking article
- `image URL`: URL leading to the fact-checked image
- `image path` : local path to the image
- `publication date` : publication date of the fact-checking article
- `was the photo used before?` : answer to the Provenance pillar
- `source` : answer to the Source pillar
- `date` : answer to the Date pillar
- `date numeric label` : answer to the Date pillar in numeric format
- `location` : answer to the Location pillar
- `motivation` : answer to the Motivation pillar

**Metadata**

The following items are provided as metadata for analysis purpose
- `org` : fact-checking organization from which the data was collected
- `claim` : claim verified in the fact-checking article
- `type of image` : the type of image claim being verified. True, Out-of-Context, Manipulated, Fake, or True
- `verification strategy` : the list of strategies used by fact-checkers to verify the image
- `verification tool` : the list of tools used by fact-checkers to verify the image
- `claimed location` : the location of the image, according to the claim
- `claimed date` : the date of the image, according to the claim


### Example

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
        "claimed date": "2021",
        "image URL": "https://factly.in/wp-content/uploads//2021/08/IAF-Airlift-Afghanistan-FB-Post.jpg"
    }

````

# Other files  🗃️

The following files can be found in the dataset folder:

- geonames_results.json: the mapping from the locations of 5Pils to GeoNames coordinates and hierarchies
- image_processing_instructions.txt: instructions to automate the cropping of the raw images collected from FC articles
- manipulation_detection_test.json: the classification results on the test set of the manipulation detection with a fine-tuned ViT transformer
- dataset/url/article_urls.txt: the URLs of the fact-checking articles from which the annotations were extracted 
- dataset/retrieval_results/evidence_url.json: the URLs of the evidence retrieved with Google Reverse Image Search

# Collect your own data 🔎

### Wayback machine scraping
We provide the code to collect FC articles and images from the 🏛️ [Wayback Machine](https://web.archive.org/) for a specific fact-checking organization. 

```
$ python scripts/data_collection_waybackmachine.py --url_domain factly.in/ --org factly --file_path dataset/url/factly.txt --parser factly_parser --scrape_image 1 --process_image 0
```

The script can be extended to cover additional fact-checking organizations. For that, you will need to create a custom scraper function and place it in dataset_collection/scrape_utils.py. If you want to process the images, e.g. by cropping them to remove social media sidebars, you will need to provide your own processing instruction file or update dataset/image_processing_instructions.txt with instructions for the new images.

Parameters:

- `url_domain`: The domain to query on the Wayback Machine API
- `org`: Organization from which articles are collected
- `file_path`: Path to the file that stores the URLs
- `parser`: A parser function. Each fact-checking organization has a dedicated one. They are found in dataset_collection/scrape_utils.py. By default, we provide parsers for Factly, Pesacheck, and 211check. If you collect data from another organization, you will need to create your own parser and add it to dataset_collection/scrape_utils.py
- `scrape_image`: If True, download the fact-checking images in addition to the article text
- `process_image`: If True, process the images by cropping them to remove social media sidebars. This requires an instruction file that is, by default, only available for the 1,676 images of 5Pils
- `image_processing_script`: Script with automated instructions to crop and clean the images. Needs to be a valid file with instructions if process_image is True. If you collect new data, you will need to update the instructions in the file.
- `start_date`: Start date for the collection of URLs
- `end_date`: End date for the collection of URLs
- `max_count`: Maximum number of URLs to collect
- `chunk_size`: Size of each chunk to query the Wayback Machine API. It is not recommended to set it higher than 5000
- `sleep`: Waiting time between two calls of the Wayback machine API. It is recommended to use a sufficiently high value. By default, 5

### Automated labeling
The next step is to automatically extract the annotations from the FC articles using a LLM. Our code uses GPT4, but can be easily modified to leverage other LLMs. Using GPT4 requires an Azure account with access to the [Azure OpenAI service](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview). You will need to provide your API key and API endpoint as parameters.

```
$ python scripts/data_labeling_gpt4.py
```

### Dataset preprocessing

The final step is to apply preprocessing functions.
The Locations  are mapped to coordinates and hierarchies using 🌍 [GeoNames](https://www.geonames.org/). You will need to create a (free) account and provide your account name as input.

```
$ python scripts/preprocess_dataset.py
```
