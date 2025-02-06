# The IEEE International Conference on Multimedia & Expo (ICME) 2025 Audio Encoder Capability Challenge

## Overview

The **ICME 2025 Audio Encoder Capability Challenge**, hosted by Xiaomi, University of Surrey, and DataOcean AI, aims to rigorously evaluate audio encoders in real-world downstream tasks.

Strongly inspired by the [HEAR benchmark](https://hearbenchmark.com/), this challenge introduces several key enhancements: a diverse task set, a focus on real-world applications, a combination of parameterized and parameter-free evaluation, and a new open-sourced, efficient evaluation system. 

**Notably, this challenge imposes NO restrictions on model size or the scale of training data, and training based on existing pre-trained models is allowed**.

Participants are invited to submit pre-trained encoders that convert raw audio waveforms into continuous embeddings. These encoders will undergo comprehensive testing across diverse tasks spanning speech, environmental sounds, and music. The evaluation will emphasize real-world usability and leverage an [open-source evaluation system](https://github.com/jimbozhang/xares).

Participants are welcome to independently test and optimize their models. However, the final rankings will be determined based on evaluations conducted by the organizers.

## How to Participate

### Registration

To participate, registration is required. Please complete the [registration form](https://forms.gle/VGgRQdPLs9f72UM8A) before **April 1, 2025**. Note that this does not means the challenge starts on April 1, 2025. The challenge begins on **February 7, 2025**.

For any other information about registration，please send Email to: <2025icme-aecc@dataoceanai.com>

### Submission

1. Clone the audio encoder template from the [GitHub repository](https://github.com/jimbozhang/xares-template.git).
2. Implement your own audio encoder following the instructions in `README.md` within the cloned repository. The implementation must pass all checks in `audio_encoder_checker.py` provided in the repository.
3. Before the submission deadline, **April 30, 2025**, email the following files to the organizers at [2025icme-aecc@dataoceanai.com](mailto:2025icme-aecc@dataoceanai.com):

- a ZIP file containing the complete repository
- a technical report paper (PDF format) not exceeding 6 pages describing your implementation

The pre-trained model weights can either be included in the ZIP file or downloaded automatically from external sources (e.g., Hugging Face) during runtime. If choosing the latter approach, please implement the automatic downloading mechanism in your encoder implementation.

While there are no strict limitations on model size, submitted models must be able to be run successfully in a Google Colab T4 environment, where the runtime is equipped with a **16 GB NVIDIA Tesla T4 GPU, 12GB RAM**.

## Datasets

This challenge emphasizes **both model innovation AND data collection and utilization**, recognizing the critical role of data in achieving superior performance.
The organizers do not mandate a specific training dataset.
Instead, participants are free to use any data for training, as long as it meets the following conditions:

- All training data must be publicly accessible. **Publicly accessible online data, including crawls, is allowed.**
- Any data in Table.1 must be excluded from the training data.
- Data derived from or augmented based on Table.1, either directly or indirectly, is not permitted for training.

Training based on existing pre-trained models, such as fine-tuning or distillation, **is allowed**, but it must be ensured that the training data of the pre-trained models does not include the data in Table.1.

### Table 1: Datasets for fine-tuning and evaluation

The ``Hidden'' column indicates whether the dataset is concealed from participants.

| **Domain** | **Dataset**               | **Task Type**                | **Metric**   | **n-classes** | **Track B** | **Hidden** |
|------------|---------------------------|------------------------------|--------------|---------------|-------------|-------------|
| **Speech** | Speech Commands           | Keyword spotting             | Acc          | 30            | ✓           | ✗           |
|            | LibriCount                | Speaker counting             | Acc          | 11            | ✓           | ✗           |
|            | VoxLingua107              | Language identification      | Acc          | 33            | ✓           | ✗           |
|            | VoxCeleb1                 | Speaker identification       | Acc          | 1251          | ✓           | ✗           |
|            | LibriSpeech               | Gender classification        | Acc          | 2             | ✓           | ✗           |
|            | Fluent Speech Commands    | Intent classification        | Acc          | 248           | ✓           | ✗           |
|            | VocalSound                | Non-speech sounds            | Acc          | 6             | ✓           | ✗           |
|            | CREMA-D                   | Emotion recognition          | Acc          | 5             | ✓           | ✗           |
|            | LibriSpeech-Phoneme       | Phoneme recognition          | Acc          | 39            | ✓           | ✗           |
|            | speechocean762            | Phoneme pronunciation        | MSE          | 3             | ✗           | ✗           |
|            | ASV2015                   | Spoofing detection           | EER          | 2             | ✓           | ✗           |
| **Sound**  | ESC-50                    | Environment classification   | Acc          | 50            | ✓           | ✗           |
|            | FSD50k                    | Sound event detection        | mAP          | 200           | ✗           | ✗           |
|            | UrbanSound 8k             | Urban sound classification   | Acc          | 10            | ✓           | ✗           |
|            | DESED                     | Sound event detection        | Segment-F1   | 10            | ✓           | ✗           |
|            | FSD18-Kaggle              | Sound event detection        | mAP          | 41            | ✗           | ✗           |
|            | Clotho                    | Sound retrieval              | Recall@1     | -             | ✗           | ✗           |
|            | Inside/outside car        | Sound event detection        | Acc          | 2             | ✓           | ✓           |
|            | Finger snap sound         | Sound event detection        | Acc          | 2             | ✓           | ✓           |
|            | Key scratching car        | Sound event detection        | Acc          | 2             | ✓           | ✓           |
|            | Subway broadcast          | Sound event detection        | Acc          | 2             | ✓           | ✓           |
|            | LiveEnv sounds            | Sound event detection        | mAP          | 18            | ✗           | ✓           |
| **Music**  | MAESTRO                   | Note classification          | Acc          | 88            | ✓           | ✗           |
|            | GTZAN Genre               | Genre classification         | Acc          | 10            | ✓           | ✗           |
|            | NSynth-Instruments        | Instruments Classification   | Acc          | 11            | ✓           | ✗           |
|            | NSynth-Pitch              | Pitches Classification       | Acc          | 128           | ✓           | ✗           |
|            | Free Music Archive Small  | Music genre classification   | Acc          | 8             | ✓           | ✗           |

**Note: The actual evaluation set selected by the organizers may include fewer datasets than those listed in Table 1.**

## Tracks

We set two tracks, Track A and Track B, is to comprehensively evaluate the performance of pre-trained audio encoders from different perspectives. Track A focuses on the adaptability and effectiveness of the pre-trained models when fine-tuned for specific tasks, while Track B assesses the inherent quality of the audio representations without any fine-tuning, providing a rigorous test of the fundamental representational power of the embeddings. **Participants do not need to choose tracks. Both tracks will be evaluated for all submissions.**

### Track A: Linear Fine-Tuning on Task-Specific Data.

A linear layer will be trained using the provided user embeddings, optimized with predefined hyperparameters for each task.
This approach assesses how effectively the fixed representations can be adapted to specific tasks by training an additional linear layer,
using predefined hyperparameters tailored for each task.
This track evaluates the adaptability and effectiveness of the pre-trained models when applied to new,
task-specific contexts without altering the original model parameters.

### Track B: Unparameterized Evaluation.

Pre-trained model embeddings will be used directly for K-nearest neighbor (KNN) classification without training.
This track aims to evaluate the inherent quality of the audio representations without any fine-tuning.
While this approach may not always yield the highest performance in real-world applications,
it serves as a rigorous test of the fundamental representational power of the embeddings.
By avoiding parameterized layers, this track provides a clear view of how well the model captures essential features of the audio data.

## Baseline

Here are the evaluation results for several baseline models. The weighted average is calculated using the test set size for each dataset.

### Table 2: Track A baseline

| Dataset                      | dasheng<br>(base) | wav2vec2<br>(large-100k-voxpopuli) | whisper<br>(base) | data2vec<br>(audio-base) |
|:----------------------------:|:---------:|:--------:|:---------:|:---------:|
| asvspoof (mini)              | **0.956** | 0.914    | 0.885     | 0.892     |
| crema_d                      | **0.772** | 0.568    | 0.600     | 0.566     |
| esc50                        | **0.869** | 0.579    | 0.614     | 0.249     |
| fluentspeechcommands_kws     | 0.916     | 0.417    | 0.878     | **0.962** |
| freemusicarchive_genre       | **0.640** | 0.518    | 0.595     | 0.360     |
| fsdkaggle2018                | **0.557** | 0.352    | 0.478     | 0.196     |
| gtzan                        | **0.869** | 0.681    | 0.751     | 0.495     |
| libricount                   | **0.688** | 0.605    | 0.549     | 0.507     |
| librispeech_male_female(mini)| 0.859     | 0.703    | **0.877** | 0.692     |
| nsynth_instument             | **0.261** | 0.251    | 0.259     | 0.223     |
| ravdess                      | **0.725** | 0.440    | 0.460     | 0.469     |
| speechcommandsv1             | **0.967** | 0.805    | 0.955     | 0.930     |
| urbansound8k                 | **0.835** | 0.676    | 0.719     | 0.443     |
| vocalsound                   | **0.910** | 0.791    | 0.871     | 0.807     |
| voxceleb1 (mini)             | **0.512** | 0.069    | 0.215     | 0.043     |
| voxlingua33 (mini)           | 0.782     | 0.492    | **0.862** | 0.577     |
| **Weighted Average**         | **0.728** | 0.500    | 0.629     | 0.541     |

---

### Table 3: Track B baseline

| Dataset                       | dasheng<br>(base) | wav2vec2<br>(large-100k-voxpopuli) | whisper<br>(base) | data2vec<br>(audio-base) |
|:-----------------------------:|:---------:|:--------:|:---------:|:---------:|
| asvspoof (mini)               | 0.833     | 0.611    | 0.600     | **0.919** |
| crema_d                       | 0.381     | 0.175    | **0.382** | 0.325     |
| esc50                         | **0.621** | 0.091    | 0.191     | 0.037     |
| fluentspeechcommands_kws      | **0.025** | 0.008    | 0.032     | 0.156     |
| freemusicarchive_genre        | **0.589** | 0.135    | 0.396     | 0.126     |
| gtzan                         | **0.753** | 0.347    | 0.504     | 0.119     |
| libricount                    | **0.310** | 0.241    | 0.253     | 0.186     |
| librispeech_male_female (mini)| 0.493     | 0.552    | 0.586     | **0.632** |
| nsynth_instument              | **0.253** | 0.235    | 0.233     | 0.209     |
| ravdess                       | **0.369** | 0.171    | 0.287     | 0.289     |
| speechcommandsv1              | **0.903** | 0.208    | 0.096     | 0.850     |
| urbansound8k                  | **0.662** | 0.334    | 0.214     | 0.153     |
| vocalsound                    | **0.336** | 0.265    | 0.417     | 0.295     |
| voxceleb1 (mini)              | **0.035** | 0.002    | 0.007     | 0.001     |
| voxlingua33 (mini)            | **0.340** | 0.014    | 0.207     | 0.050     |
| **Weighted Average**          | **0.384** | 0.271    | 0.251     | 0.350     |


## Important Dates

- **February 7, 2025**: Challenge announcement and start.
- **April 1, 2025**: Registration deadline ([Register here](https://forms.gle/VGgRQdPLs9f72UM8A)).  
- **April 30, 2025**: Submission deadline.  
- **May 27, 2025**: Results announced.

## More Details

The more detailed description of the challenge can be found in the [ICME 2025 Audio Encoder Capability Challenge](https://arxiv.org/abs/2501.15302) paper.
