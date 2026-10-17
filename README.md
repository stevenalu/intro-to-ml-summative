# Intro to ML - Summative

Predicting student final outcomes on Open University online courses using traditional machine learning and deep learning.

The notebook trains eight models on the Open University Learning Analytics Dataset (OULAD) and compares them on accuracy, macro F1, and macro ROC-AUC.

## Author

Principie Cyubahiro, Bachelor of Software Engineering, African Leadership University.

## Problem

Online learning platforms struggle to spot students who will fail or drop out before it is too late. The target in this project is the `final_result` column: Pass, Fail, Withdrawn, or Distinction. The task is four-class classification on 32,593 student-course enrolments.

## Dataset

Open University Learning Analytics Dataset (OULAD).

- Kaggle: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad
- Original paper: Kuzilek, Hlosta, and Zdrahal (2017), *Scientific Data*, DOI 10.1038/sdata.2017.171
- License: CC BY 4.0

The raw CSVs are not stored in this repository. Download them from Kaggle and place them in a local folder (not tracked by git). The full raw dataset and the saved model weights live in the Google Drive folder linked below.

Files used by the notebook:
- studentInfo.csv
- studentAssessment.csv
- assessments.csv
- studentVle.csv
- vle.csv

## How to run

Clone the repo and install dependencies:

```
git clone https://github.com/stevenalu/intro-to-ml-summative.git
cd intro-to-ml-summative
pip install -r requirements.txt
```

Download the OULAD files from the Kaggle link above and put them in a folder exactly named like that used in the notebook, for example `oulad/`.

Open the notebook in Jupyter or Colab:

```
jupyter notebook Summative_Model_Training_and_Evaluation_Intro_to_ML.ipynb
```

In Colab, mount Drive first (there is a cell for this already) and set `DATA_DIR` to your Drive path, for example `/content/drive/MyDrive/oulad`.

Run top to bottom. Random seed is 42 so the numbers below are reproducible.

## Experiments

Eight experiments across two paradigms:

Traditional machine learning (Scikit-Learn):
1. Logistic Regression
2. Decision Tree
3. Random Forest (defaults)
4. Random Forest tuned
5. XGBoost

Deep learning (TensorFlow Keras):
6. Multilayer Perceptron (MLP)
7. 1D-CNN
8. 1D-CNN tuned

## Results

Test set, 6,519 rows.

| Exp | Model | Accuracy | Macro F1 | Macro ROC-AUC |
|-----|-------|----------|----------|---------------|
| 1 | Logistic Regression | 0.7398 | 0.6745 | 0.9151 |
| 2 | Decision Tree | 0.6963 | 0.6498 | 0.8458 |
| 3 | Random Forest | 0.7466 | 0.6803 | 0.9223 |
| 4 | Random Forest tuned | 0.7467 | 0.6804 | 0.9240 |
| 5 | XGBoost | 0.7595 | 0.7064 | 0.9297 |
| 6 | MLP | 0.7389 | 0.6756 | 0.9196 |
| 7 | 1D-CNN | 0.6992 | 0.5527 | 0.8895 |
| 8 | 1D-CNN tuned | 0.5960 | 0.3713 | 0.8490 |

**Best model: XGBoost** at 75.95 percent accuracy. This sits inside the 75 to 82 percent range reported in the OULAD literature for four-class prediction.

## Project structure

```
.
├── Summative_Model_Training_and_Evaluation_Intro_to_ML.ipynb   # main notebook
├── README.md
├── requirements.txt
```

Not in the repository (kept in the Google Drive folder):
- raw CSV files
- saved model weights (`.pkl` for the best traditional model, `.keras` for MLP and CNN)

## Links

- Google Drive folder (raw data + saved models): [https://drive.google.com/drive/folders/1XujFV3ZWCmZS7HO7w4Eqi6KN99PqDHUY?usp=sharing](https://drive.google.com/drive/folders/1XujFV3ZWCmZS7HO7w4Eqi6KN99PqDHUY?usp=sharing)
- Demo Video: [Demo Video](https://drive.google.com/file/d/1rmNNsBFqg7P-tMn6KumTNmLMFSYdRgDy/view?usp=sharing)

## Notes

- Only numeric features whose values would naturally be zero when absent are filled with zero (clicks, scores for students with no activity). The one categorical column with missing values, `imd_band`, is filled with the explicit category `Unknown`.
- Random seed 42 is set in both NumPy and TensorFlow before any model is instantiated.
- The 1D-CNN is included for comparison on identical inputs. Tabular features have no spatial order, so convolution has no local structure to exploit, which is why the CNN underperforms. This is an expected result and is discussed in the report.
