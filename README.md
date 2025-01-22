# Challenge Title

## Abstract

This challenge aims to establish a benchmark for evaluating the capabilities of audio encoders, especially in the context of multi-task learning and real-world applications. Participants are invited to submit pre-trained audio encoders that map raw waveforms to continuous embeddings. These encoders will be assessed across a diverse range of tasks spanning voice, environmental sounds, and music, with a focus on their effectiveness as components in larger systems. The challenge features two tracks: Track A for parameterized evaluation, and Track B for parameter-free evaluation. To emphasize practical relevance, the organizers are releasing several novel, open-source datasets featuring diverse audio recordings from real-world scenarios and users. This challenge provides a critical platform for benchmarking and advancing the state-of-the-art in audio encoder design.

## Challenge Description

The field of audio representation learning has advanced significantly in recent years, enabling models to extract meaningful features from audio data effectively. While much of the focus has been on discrete representations and tokenization, continuous representations remain crucial for many tasks. Unlike discrete tokens, continuous embeddings retain the nuanced information within audio signals, leading to better performance in downstream tasks like fine-grained classification, regression, and time-series analysis. Moreover, continuous audio encoders play a key role in multimodal large language models, facilitating the integration of audio with other modalities. Models such as wav2vec2, Data2vec2, and Dasheng have demonstrated strong performance across various audio tasks. While there are some existing benchmarks for evaluating these models, they leave room for further refinement, and a comparison with similar benchmarks can be found in Appendix A.

This challenge provides a platform for participants to showcase innovative approaches in model design and data utilization, pushing the boundaries of audio representation learning. Participants are required to submit a single pre-trained encoder that processes audio waveforms and generates two outputs: a sequence of continuous embedding vectors for frame-level tasks and a fixed-dimension embedding vector for utterance-level tasks. The model should comply with an API specified by the organizers, with examples provided.

The submitted models will be evaluated on diverse audio tasks, including human voice, environmental sounds, and music, using an open-source evaluation system. Participants can test and optimize models independently, but final rankings will be based on evaluations conducted on the organizers’ servers. This challenge aims to advance the state-of-the-art in continuous audio representation learning.

### Tracks

The challenge consists of two tracks, each evaluating the pre-trained models in different ways.

**Track A: Linear Fine-Tuning on Task-Specific Data.** A linear layer will be trained using the provided user embeddings, optimized with predefined hyperparameters for each task. This approach assesses how effectively the fixed representations can be adapted to specific tasks by training an additional linear layer, using predefined hyperparameters tailored for each task. This task evaluates the adaptability and effectiveness of the pre-trained models when applied to new, task-specific contexts without altering the original model parameters.

**Track B: Unparameterized Evaluation.** Pre-trained model embeddings will be used directly for K-nearest neighbor (KNN) classification without training. This track aims to evaluate the inherent quality of the audio representations without any fine-tuning. While this approach may not always yield the highest performance in real-world applications, it serves as a rigorous test of the fundamental representational power of the embeddings. By avoiding parameterized layers, this track provides a clear view of how well the model captures essential features of the audio data.

### Table 1: Proposed Benchmark Datasets

| **Domain**      | **Dataset**                                             | **Task Type**                | **Metric**   | **n-classes** | **Track B** |
|-----------------|---------------------------------------------------------|------------------------------|--------------|---------------|-------------|
| **Speech**      | Speech Commands                                         | Keyword spotting             | Acc          | 30            | ✓           |
|                 | LibriCount                                              | Speaker counting             | Acc          | 11            | ✓           |
|                 | VoxLingua107                                            | Language identification      | Acc          | 33            | ✓           |
|                 | VoxCeleb1                                               | Speaker identification       | Acc          | 1251          | ✓           |
|                 | LibriSpeech                                             | Gender classification        | Acc          | 2             | ✓           |
|                 | Fluent Speech Commands                                  | Intent classification        | Acc          | 248           | ✓           |
|                 | VocalSound                                              | Non-speech sounds            | Acc          | 6             | ✓           |
|                 | CREMA-D                                                 | Emotion recognition          | Acc          | 5             | ✓           |
|                 | LibriSpeech-Phoneme                                     | Phoneme recognition          | Acc          | 39            | ✓           |
|                 | speechocean762                                          | Phoneme pronunciation        | MSE          | 3             | ✗           |
|                 | ASV2015                                                 | Spoofing detection           | EER          | 2             | ✓           |
| **Sound**       | ESC-50                                                  | Environment classification   | Acc          | 50            | ✓           |
|                 | FSD50k                                                  | Sound event detection        | mAP          | 200           | ✗           |
|                 | UrbanSound 8k                                           | Urban sound classification   | Acc          | 10            | ✓           |
|                 | DESED                                                   | Sound event detection        | Segment-F1   | 10            | ✓           |
|                 | FSD18-Kaggle                                            | Sound event detection        | mAP          | 41            | ✗           |
|                 | Clotho                                                  | Sound retrieval              | Recall@1     | -             | ✗           |
|                 | Inside/outside car†                                     | Sound event detection        | Acc          | 2             | ✓           |
|                 | Finger snap sound†                                      | Sound event detection        | Acc          | 2             | ✓           |
|                 | Key scratching car†                                     | Sound event detection        | Acc          | 2             | ✓           |
|                 | Subway broadcast†                                       | Sound event detection        | Acc          | 2             | ✓           |
|                 | RealUser sounds†                                        | Sound event detection        | mAP          | 14            | ✗           |
|                 | LiveEnv sounds†                                         | Sound event detection        | mAP          | 18            | ✗           |
| **Music**       | MAESTRO                                                 | Note classification          | Acc          | 88            | ✓           |
|                 | GTZAN Genre                                             | Genre classification         | Acc          | 10            | ✓           |
|                 | NSynth-Instruments                                      | Instruments Classification   | Acc          | 11            | ✓           |
|                 | NSynth-Pitch                                            | Pitches Classification       | Acc          | 128           | ✓           |
|                 | Free Music Archive Small                                | Music genre classification   | Acc          | 8             | ✓           |

### Training Dataset

The challenge places a significant emphasis on data collection and utilization, which is a crucial component of the competition. The organizers do not prescribe a specific training dataset. Instead, participants are free to use any data for training, as long as it meets the following conditions:

- All training data must be publicly accessible.
- Data in Table 1 must be excluded from training.
- Data derived from or augmented based on the test set, either directly or indirectly, is not permitted for training.

### Datasets for Fine-tuning and Evaluation

The datasets, outlined in Table 1, comprise a diverse range of audio data spanning multiple domains, including human voice, environmental sounds, and music. We utilize each dataset's native train-test split to fine-tune and test participant-submitted models. All datasets are open-sourced, with six new datasets focusing on real-world industrial scenarios provided by the challenge organizers themselves, marked with † in Table 1.

Table 2 provides an overview of the datasets introduced by the challenge organizers. These datasets are designed to reflect real-world industrial scenarios, enriching the diversity and practical applicability of the challenge. Note that a portion of these six datasets will have their ground truth labels hidden to participants.

### Table 2: Datasets Released by the Challenge Organizers

| **Dataset**           | **Size**  | **Description** | **Hidden**  |
|-----------------------|-----------|-----------------|-------------|
| Inside/outside car     | 15k samples | Security threat prevention by distinguishing environments | ✗ |
| Finger snap sound      | 15k samples | Wake-up word alternative for smart speakers               | ✗ |
| Key scratching car     | 5k samples  | Detecting car vandalism through key scratching sounds     | ✗ |
| Subway broadcast       | 125 hours  | Noise canceling in response to subway announcements       | ✓ |
| RealUser sound         | 105k samples | Diverse real-world sound events captured by users         | ✓ |
| LiveEnv sound          | 25k samples  | Authentic environmental sounds from various locations     | ✓ |

## Submission Guide

Participants are required to submit a pre-trained model encapsulated within the specified API. The model should accept a single-channel audio signal, represented as a PyTorch tensor with shape [B, T], where B denotes the batch size and T represents the number of samples in the time domain. The model should output a frame-level prediction of shape [B, T', D], where T' can be different from the input T and D is the embedding dimension defined by the participant.

While there are no strict limitations on model size, submitted models must be able to be run successfully in a Google Colab T4 environment, where the runtime is equipped with a 16 GB NVIDIA Tesla T4 GPU, 12GB RAM.

Participants are also required to submit a technical report along with their submission.

The submission steps are as follows:

1. Clone the audio encoder template from the GitHub repository.
2. Implement your own audio encoder following the instructions in `README.md` within the cloned repository. The implementation must pass all checks in `audio_encoder_checker.py` provided in the repository.
3. Before the submission deadline, email the organizers a ZIP file containing the complete repository. In the body of the email, please include the names, email addresses, and affiliations of all team members. Additionally, please attach a technical report paper (PDF format) not exceeding 6 pages describing your implementation. Pre-trained model weights can either be included in the ZIP file or downloaded automatically from external sources (e.g., HuggingFace) during runtime. If choosing the latter approach, please implement the automatic downloading mechanism in your encoder implementation.

## Evaluation and Ranking

The performance metrics for each task are normalized to a 0-1 scale, and the final score is computed based on these normalized metrics.

### Normalization of Metrics

Each task in Table 1, i.e. Ti, has an associated metric Mi (e.g., accuracy, EER, mAP, F1). To normalize these metrics, we use the following formula:

\[
\hat{M}_i = \frac{M_i - M_i^{\text{min}}}{M_i^{\text{max}} - M_i^{\text{min}}}
\]

where \(\hat{M}_i\) is the normalized metric for task \(T_i\), and \(M_i\) is the raw metric value for task \(T_i\). \(M_i^{\text{min}}\) and \(M_i^{\text{max}}\) are the minimum and maximum possible values of the metric \(M_i\), respectively.

For instance, the accuracy, EER, and F1 scores range from 0 to 1, so their \(M_i^{\text{min}}\) and \(M_i^{\text{max}}\) are 0 and 1, respectively; mAP ranges from 0 to 100, so for mAP tasks, \(M_i^{\text{min}} = 0\) and \(M_i^{\text{max}} = 100\).

### Final Score and Ranking

The final score for each participant for Track A and Track B is calculated as the weighted average of the normalized metrics across all tasks applicable to the respective task, where the weight is determined by the size of the test set for each task. This approach ensures that tasks with larger test sets have a greater impact on the final score, reflecting their significance in evaluating the model's performance. The final scores \(S_A\) and \(S_B\) for Track A and Track B are given by:

\[
S_{\text{track}} = \frac{\sum_{i=1}^{N_{\text{task}}} n_i \hat{M}_i}{\sum_{i=1}^{N_{\text{task}}} n_i}
\]

where \(N_{\text{task}}\) is the total number of tasks applicable to the respective task, \(n_i\) is the size of the test set for task \(T_i\), and \(\hat{M}_i\) is the normalized metric for task \(T_i\).

Participants are ranked within each track based on their final scores, \(S_A\) and \(S_B\), respectively. The overall performance of the participants will be showcased in two separate leaderboards, one for Track A and one for Track B, to accurately reflect competencies in both parameterized and unparameterized evaluation methodologies.

## Challenge Organizers

This Challenge is organized by teams from three institutions: Xiaomi Corporation, the University of Surrey and Dataocean AI Inc.

**Xiaomi Corporation** is a renowned technology company established in 2010. It is widely known for its diverse product range including smartphones, cars, tablets, laptops, wearables, and smart home devices, to form a platform of more than 800 million active devices. The company emphasizes innovation and user experience, is dedicated to fundamental technologies, blends into open-source. AI has been fully integrated into to reinforce Xiaomi's machie intelligence and service efficiency, ranging from user interaction, imaging, auto pilot, to internet sales, delivery, and service. Among them, the acoustic and speech team of the AI lab is committed to us large audio and speech models to boost the research and development in speech recognition, speech synthesis, microphone array based noise reduction, voice trigger, extraction and understanding of rich language, and acoustic measurement.

**Dr. Junbo Zhang** is an expert in acoustic and speech technology at Xiaomi. He earned his Ph.D. from the Institute of Acoustics at the Chinese Academy of Sciences. With years of experience in the development of acoustic and speech algorithms, Dr. Zhang has made contributions to various fields including speech recognition, pronunciation evaluation, speech synthesis, audio tagging, sound separation, and noise reduction. He has authored over 30 papers in prestigious journals and top-tier conferences. As a code contributor to the open-source project Kaldi, he also wrote the book "Kaldi Speech Recognition Practice", which has sold more than ten thousand copies. At Xiaomi, he was instrumental in developing and launching the company's initial speech recognition system, the wake word detection for "Xiao Ai" (Xiaomi's AI assistant), and the voiceprint recognition system. Currently, he leads several pioneering projects in the sound technology domain, pushing the boundaries of what's possible in consumer electronics with voice interaction.

**University of Surrey** The Machine Audition Lab within the Centre for Vision Speech and Signal Processing at the University of Surrey, led by Prof Wenwu Wang, is a leading research lab in audio signal processing and machine learning, consisting more than 30 researchers. They have developed several widely used audio representation models such as PANNs, AudioLDM, AudioLDM 2, AudioSep, etc. They have been contributing to the activities in Detection and Classification of Acoustic Scenes and Events (DCASE) challenges and workshops since 2013, including the organisation of two tasks of the DCASE 2024 Challenges, i.e. Task 6 - Automated Audio Captioning and Task 9 - Language-Queried Audio Source Separation.

**Dr. Wenwu Wang** is a Professor in Signal Processing and Machine Learning, University of Surrey, UK. He is also an AI Fellow at the Surrey Institute for People Centred Artificial Intelligence. His current research interests include signal processing, machine learning and perception, artificial intelligence, machine audition (listening), and statistical anomaly detection. He has (co)-authored over 300 papers in these areas. He has been recognized as a (co-)author or (co)-recipient of more than 15 accolades, including the 2022 IEEE Signal Processing Society Young Author Best Paper Award, ICAUS 2021 Best Paper Award, DCASE 2020 and 2023 Judge’s Award, DCASE 2019 and 2020 Reproducible System Award, and LVA/ICA 2018 Best Student Paper Award. He is an Associate Editor (2020-2025) for IEEE/ACM Transactions on Audio Speech and Language Processing, and an Associate Editor (2024-2026) for IEEE Transactions on Multimedia. He was a Senior Area Editor (2019-2023) and Associate Editor (2014-2018) for IEEE Transactions on Signal Processing. He is the elected Chair (2023-2024) of IEEE Signal Processing Society (SPS) Machine Learning for Signal Processing Technical Committee, a Board Member (2023-2024) of IEEE SPS Technical Directions Board, the elected Chair (2025-2027) and Vice Chair (2022-2024) of the EURASIP Technical Area Committee on Acoustic Speech and Music Signal Processing, an elected Member (2021-2026) of the IEEE SPS Signal Processing Theory and Methods Technical Committee. He has been on the organising committee of INTERSPEECH 2022, IEEE ICASSP 2019 & 2024, IEEE MLSP 2013 & 2024, and SSP 2009. He is Technical Program Co-Chair of IEEE MLSP 2025. He has been an invited Keynote or Plenary Speaker on more than 20 international conferences and workshops.

**Dataocean AI Inc.**, founded in 2005, is a global professional service provider specializing in AI training data. The enterprise is dedicated to delivering data products and solutions across core AI fields, including Text-to-Speech, Automatic Speech Recognition, Natural Language Processing, Lexicon, Computer Vision, and Multi-modal technologies.

## Challenge Schedule

The Challenge will follow this schedule:

- February 7, 2025: Challenge announcement and release of evaluation details.
- April 30, 2025: Submission deadline.
- May 27, 2025: Results announcement.

## Related Work

Our challenge broadens the scope by including non-speech related tasks, enabling a more comprehensive evaluation of audio encoders, which are pivotal in both continuous and discrete audio processing contexts. Here, we discuss three of the existing benchmarks, highlighting the unique contributions and improvements of our proposed challenge.

### HEAR: Holistic Evaluation of Audio Representations

Our proposed challenge is strongly inspired by the HEAR benchmark, which assesses audio representations across environmental sound and music tasks. While HEAR provides an excellent foundation, our challenge introduces several enhancements:

**Diverse task set:** HEAR comprises 19 tasks in total, 17 of which are unique, while two tasks differ only in their available training data. While the tasks in HEAR encompass various application scenarios for sound event detection and music processing, they lack variety in human voice processing. Our challenge offers a more comprehensive and balanced distribution of tasks across human voice, music, and environmental sound domains, leveraging a suite of open-source datasets that reflect real-world scenarios and user experiences, including unique datasets such as car scratching, inside/outside car environments, and user-generated sounds. Specifically, our challenge includes 12 tasks related to human voice, 12 tasks related to environmental sounds, and 5 tasks related to music.

**Focus on real-world applications:** Some tasks in HEAR, although interesting, may have limited applications and high variance during testing (e.g., Gunshot Triangulation and Beehive) due to the factors such as small sample sizes, which have led to many follow-up works discarding those tasks. We seek to balance task variety, real-world impact, and robust performance estimation, ensuring the evaluated representations are relevant to industrial use, providing reliable performance metrics. Furthermore, we are releasing a collection of open-source datasets, featuring a diverse range of audio recordings from real-world scenarios, including in-car and outside-car environments, finger snap sounds, key scratching car, subway broadcast, and user-generated data, all of which are carefully curated to reflect practical and scenario-oriented applications.

**Evaluation methods:** In addition to linear projection, we utilize unparameterized methods for classification. This evaluation aims at investigating the use of features for cases such as unsupervised clustering.

**Efficient system:** We propose an open-sourced, efficient evaluation system, incorperating a simple pipline that can be run without any prequisites.

### SUPERB: Speech processing Universal PERformance Benchmark

SUPERB and its derivatives primarily focus on speech processing tasks using self-supervised learning (SSL) representations. In recent years, SUPERB also included additional tasks such as emotion recognition and sound codecs, but notably, it does not include environmental audio or music related tasks.

Our challenge broadens this scope with the inclusion of non-speech related tasks (environmental audio, music), enabling a more comprehensive evaluation of audio representations.

### DASB: Discrete Audio and Speech Benchmark

DASB benchmarks discrete audio tokens across various tasks, mainly focuses on the speech domain. While discretization is an important research field, continuous representations offer complementary advantages. Continuous representations often achieve better performance in tasks requiring fine-grained distinctions, and they are directly compatible with many existing machine learning architectures and pipelines, unlike discrete tokens which often require specific handling. In addition, the focus on continuous representations directly addresses the need for robust audio encoders in multimodal applications, where continuous embeddings are often preferred for seamless integration and efficient processing. The output of this challenge can be used to complement discrete representation research by, for example, injecting general semantic information into codecs, or evaluating the loss of information during the discretisation process. Our challenge prioritizes established methods for continuous representations.
