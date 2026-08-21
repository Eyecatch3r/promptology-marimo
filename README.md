# Promptology:  Large-Scale Systematic Analysis of Prompt Elements and Contextual Factors in Argument Generation
This repository contains the code and data for the paper "Promptology: Large-Scale Systematic Analysis of Prompt Elements and Contextual Factors in Argument Generation" by Maximilian Maurer, Ana Lisboa, Matteo Melis, Julia Romberg, Aldo Costa and Gabriella Lapesa, to appear in EMNLP 2026.

# Dataset
To access the LLM-generated arguments, please refer to [https://huggingface.co/datasets/mmmaurer/promptology](https://huggingface.co/datasets/mmmaurer/promptology). The dataset is licensed under the Creative Commons Attribution NonCommercial 4.0 International License (CC BY-NC 4.0).

# Raw Results
For full raw results of the experiments, please refer to [https://huggingface.co/datasets/mmmaurer/promptology-results](https://huggingface.co/datasets/mmmaurer/promptology-results). The dataset is licensed under the Creative Commons Attribution NonCommercial 4.0 International License (CC BY-NC 4.0).

# Repository Structure
The repository is organized as follows:

- `src/`: Contains code for prompt templates and utils, and scripts for prompt prompt generation, data gathering (API prompt calls), feature extraction, and diagnostic classification experiments.
- `figures/`: Contains figures used in the paper.
- `prompts/`: Contains the prompts used in the experiments.

In the base directory, you will find the following organizationfiles:
- `README.md`: This file, providing an overview of the repository and instructions for use.
- `pyproject.toml`: The configuration file for the Python project, specifying dependencies and other settings.


# Reproduction Instructions
## Requirements
The code was developed and tested using Python 3.13. The required packages are specified in `pyproject.toml`. You can install the dependencies using `uv` or `pip`:

With `uv`:
```bash
uv sync
```

With `pip`:
```bash
pip install -e .
```

Since we use the ``elfen``package for linguistic feature extraction, you will need to install additional resources for it. Please follow the instructions in the [elfen repository](https://github.com/mmmaurer/elfen).

For inference, you will also need access to some API providing GPT-4o-mini and LLaMA 4 scout. 

## Data Preparation and Analysis
For full reproduction, you will need to download the PerspectiveArg dataset and the smartvote questionnaire data. The PerspectiveArg dataset can be found [here](https://github.com/Blubberli/perspective-argument-retrieval). The smartvote questionnaire data has to be requested from smartvote. Please contact the smartvote team via [https://www.smartvote.ch/en/contact](https://www.smartvote.ch/en/contact) to request access to the data. For any questions or problems regarding the data, please contact us via email.

In a first step, we merge the human-written arguments from the PerspectiveArg dataset with the questionnaire data from smartvote:
- `merge_datasets.ipynb`

## Prompt Generation and Cleanup
We generate the prompts, gather the outputs (see `src/` for the code), do inference and clean and streamline the outputs for analysis:
- `inference_llama-occiglot.ipynb` for LLaMA 3 and Occiglot.
- `join_occiglot_chunks.ipynb`
- `cleanup_gpt.ipynb`
- `cleanup_occiglot.ipynb`
- `add_ids.ipynb`
- `transform_llama3.ipynb`

Note that inference on LLaMA 4 scout and GPT-4o-mini is not fully reproducible just from the code, as it was run on a local OpenWebUI API. However, we provide the outputs of these models in the dataset and the script for the calls to the APIs in `src/internal_api.py`. You can use this script to run inference on your own, if you have access to an API.

## Feature Extraction
We extract linguistic features from the outputs (and the human-written arguments):
- `src/feature_extraction.py`
and prepare the data for the analyses:
- `preprocessing.ipynb`

## Analysis
We perform the diagnostic classification experiments (difference analysis) and stability analyses:
- `src/rq1_sbert.py` for the classification experiments using SBERT features.
- `src/rq1_linguistic-features.py` for the classification experiments using linguistic features.
- `Similarity_Analysis.ipynb` for Rouge-L scores.
- `bert_score_analysis.ipynb` for BERTScores.
- `lmm.ipynb` for the difference and stability analysis regression models.
And finally, we prepare data for qualitative analysis:
- `error_analysis.ipynb`

# Citation
If you use the data, analysis framework, or any code from this repository, please cite the paper as follows:

```
@inproceedings{maurer2026promptology,
  title={Promptology: Large-Scale Systematic Analysis of Prompt Elements and Contextual Factors in Argument Generation},
  author={Maximilian Maurer and Ana Lisboa and Matteo Melis and Julia Romberg and Aldo Costa and Gabriella Lapesa},
  booktitle={To appear: Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026}
}

