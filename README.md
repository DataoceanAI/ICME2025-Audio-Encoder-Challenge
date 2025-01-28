# The IEEE International Conference on Multimedia & Expo (ICME) 2025 Audio Encoder Capability Challenge

## Overview

The **ICME 2025 Audio Encoder Capability Challenge** aims to evaluate audio encoders in multi-task learning and real-world applications. It is strongly inspired by the [HEAR benchmark](https://hearbenchmark.com/), with several enhancements introduced: diverse task set, real-world applications, both of parameterized evaluation and parameter-free evaluation, and an new open-sourced, efficient evaluation system.

Participants submit pre-trained encoders that convert raw audio waveforms into continuous embeddings. These encoders will be tested across diverse tasks including **speech, environmental sounds, and music**, with a focus on real-world usability, using an [open-source evaluation system](https://github.com/jimbozhang/xares).
Participants may test and optimize models independently, but the final rankings will be based on the evaluations by the organizers. This challenge aims to advance the state-of-the-art in continuous audio representation learning.

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

While there are no strict limitations on model size, submitted models must be able to be run successfully in a Google Colab T4 environment, where the runtime is equipped with a 16 GB NVIDIA Tesla T4 GPU, 12GB RAM.

## Datasets

The challenge places a significant emphasis on data collection and utilization, which is a crucial component of the competition.
The organizers do not prescribe a specific training dataset.
Instead, participants are free to use any data for training, as long as it meets the following conditions:

- All training data must be publicly accessible.
- Any data in Table.1 must be excluded from the training data.
- Data derived from or augmented based on Table.1, either directly or indirectly, is not permitted for training.

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


## Tracks

We set two tracks, Track A and Track B, is to comprehensively evaluate the performance of pre-trained audio encoders from different perspectives. Track A focuses on the adaptability and effectiveness of the pre-trained models when fine-tuned for specific tasks, while Track B assesses the inherent quality of the audio representations without any fine-tuning, providing a rigorous test of the fundamental representational power of the embeddings. **Participants do not need to choose tracks. Both tracks will be evaluated for all submissions.**

### Track A: Linear Fine-Tuning on Task-Specific Data.

A linear layer will be trained using the provided user embeddings, optimized with predefined hyperparameters for each task.
This approach assesses how effectively the fixed representations can be adapted to specific tasks by training an additional linear layer,
using predefined hyperparameters tailored for each task.
This task evaluates the adaptability and effectiveness of the pre-trained models when applied to new,
task-specific contexts without altering the original model parameters.

### Track B: Unparameterized Evaluation.

Pre-trained model embeddings will be used directly for K-nearest neighbor (KNN) classification without training.
This track aims to evaluate the inherent quality of the audio representations without any fine-tuning.
While this approach may not always yield the highest performance in real-world applications,
it serves as a rigorous test of the fundamental representational power of the embeddings.
By avoiding parameterized layers, this track provides a clear view of how well the model captures essential features of the audio data.

## Important Dates

- **February 7, 2025**: Challenge announcement and start.
- **April 1, 2025**: Registration deadline ([Register here](https://forms.gle/VGgRQdPLs9f72UM8A)).  
- **April 30, 2025**: Submission deadline.  
- **May 27, 2025**: Results announced.

## More Details

The more detailed description of the challenge can be found in the [ICME 2025 Audio Encoder Capability Challenge](https://arxiv.org/abs/2501.15302) paper.
