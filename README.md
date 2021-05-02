# Signo-Lingo

Signo Lingo is a video classification project that classifies sign languages into English semantic meaning. The model is trained on the Turkish Sign Language dataset (AUTSL).

## 1. Table of contents <a name="TOC"></a>

1. [Table of contents](#TOC)
2. [Directory structure](#DS)
3. [Problem Statement](#PS)

## 2. Directory structure <a name="DS"></a>

```utf-8
.
├── src/                                   # logs gathered from 
│   ├── dataset/                           # stores train,val,test dataset
│   ├── dev_model/                         # stores model
│   ├── test_model/                        # store notebook for testing
│   ├── train_final/                       # final notebook
│   └── Exploratory Data Analysis.ipynb    # notebook to run data analysis
├── models/                                # saved models
├── web_gui/                               # code for web GUI
├── Big_Project_Instructions.pdf           # instructions for project
└── README.md
```

## 3. Problem Statement <a name="PS"></a>

Sign language is a way of expressing oneself primarily through a series of hand gestures. However, it is dominantly used by the deaf and mute community and may prove to be a challenge for others who do not use the language to understand. Moreover, sign language is not universal -  each community develops its own sign language with unique intricacies. As such, this project aims to develop a sign language video classification model that will help others to interpret the semantic meanings behind the language.

**Inputs**: Videos of gestures representing words in the Turkish Sign Language.
**Outputs**: Classes that are mapped to a specific meaning, in English.
**Deliverables**: Source Code (models, weights, logs, notebooks) and GUI
