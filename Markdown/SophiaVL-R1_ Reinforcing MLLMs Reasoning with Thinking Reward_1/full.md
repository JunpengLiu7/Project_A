# SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward  

Kaixuan $\mathbf { F a n ^ { 1 , 2 * \dagger } } ,$ Kaituo Fengl\*, Haoming Lyu?, Dongzhan Zhou?t, Xiangyu Yue 'MMLab, The Chinese University of Hong Kong2Shanghai Artifcial Intelligence Laboratory https://github.com/kxfan2002/SophiaVL-R1  

# Abstract  

Recent advances have shown success in eliciting strong reasoning abilities in multimodal large language models (MLLMs) through rule-based reinforcement learning (RL) with outcome rewards. However, this paradigm typically lacks supervision over the thinking process leading to the final outcome. As a result, the model may learn sub-optimal reasoning strategies, which can hinder its generalization ability. In light of this, we propose SophiaVL-R1, as an attempt to add reward signals for the thinking process in this paradigm. To achieve this, we first train a thinking reward model that evaluates the quality of the entire thinking process. Given that the thinking reward may be unreliable for certain samples due to reward hacking, we propose the Trust-GRPO method, which assigns a trustworthiness weight to the thinking reward during training. This weight is computed based on the thinking reward comparison of responses leading to correct answers versus incorrect answers, helping to mitigate the impact of potentially unreliable thinking rewards. Moreover, we design an annealing training strategy that gradually reduces the thinking reward over time, allowing the model to rely more on the accurate rule-based outcome reward in later training stages. Experiments show that our SophiaVL-R1 surpasses a series of reasoning MLLMs on various benchmarks (e.g., MathVisita, MMMU), demonstrating strong reasoning and generalization capabilities. Notably, our SophiaVL-R1-7B even outperforms LLaVA-OneVision-72B on most benchmarks, despite the latter having $1 0 \times$ more parameters. All code, models, and datasets are made publicly available at https://github.com/kxfan2002/SophiaVL-R1.  

# 1 Introduction  

Recent advances have highlighted the potential of rule-based Reinforcement Learning (RL) to elicit reasoning capabilities of Large Language Models (LLMs) [9, 37]. In particular, DeepSeek-R1 [9] exemplifies the success of applying the GRPO [22] reinforcement learning algorithm to incentive strong reasoning with long Chain-of-Thought (CoT) in LLMs. Beyond text-based domains, this paradigm has also shown promising results in Multimodal Large Language Models (MLLMs), with representative models including R1-OneVision [34], OpenVLThinker [5], and Video-R1 [7]. The key of these methods is to utilize a rule-based function that yields accurate outcome reward signals for RL training [9, 11, 5].  

However, solely relying on the outcome reward usually fails to ensure the quality of the thinking process, which is critical for developing models with generalizable reasoning ability [16]. For example, models may produce correct answers through flawed thinking trajectories, as illustrated in Figure 1, rather than through systematic deduction. During GRPO training [22], the rule-based outcome reward will equally encourage these responses with correct answers, regardless of whether the underlying thinking process is sound or flawed. Therefore, the model may adopt sub-optimal or even wrong reasoning strategies that generalize poorly, leading to inferior performance. This gives rise to one intuitive thought: Can we incorporate a reward for the thinking process during GRPO training to explicitly guide correct reasoning?  

![](images/6412f663f5a4064f2dd6eb35f51f7b9b52c8dc98f7b41c16fcc2f41b45c55ec9.jpg)  
Figure 1: Examples of model responses and their corresponding thinking rewards.  

To explore this question, we propose SophiaVL-R1, an MLLM that enhances reasoning by integrating model-generated thinking rewards with rule-based outcome rewards in RL training. Given that typical process reward models (PRMs) impose rigid step-wise constraints on reasoning and can be overly exploited (e.g., generating meaningless or repetitive steps), we measure the quality of the thinking process at a holistic level rather than at the step level. Specifically, we introduce a thinking reward model trained on annotated reasoning responses collected from GRPO training trajectories. This model evaluates intermediate reasoning quality based on criteria such as logical soundness, consistency across steps, and redundancy in the thinking process. By doing so, we provide reward signals that help the reasoning model distinguish between sound and flawed thinking processes.  

Moreover, considering that the model-generated thinking rewards may be unreliable for certain cases [36, 13], we propose the Trust-GRPO training algorithm to reduce the risks of reward hacking [26]. The core idea of Trust-GRPO is to add a trustworthiness weight to the thinking reward, which evaluates the reliability of the rewards across a group of responses to a given question. This weight is determined by comparing the thinking rewards of responses that produce correct answers with those that yield incorrect answers for the same question. A lower trustworthiness weight is assigned when high thinking rewards are abnormally given to reasoning processes that lead to incorrect answers, indicating that the reward signal may be unreliable. Unlike previous uncertainty estimation methods such as MC Dropout [8], which usually require multiple samplings for a single response--an approach that can be computationally prohibitive for MLLMs--our method introduces no additional cost by leveraging information from the response group within GRPO. Furthermore, an annealing schedule is introduced to gradually reduce the influence of the thinking reward throughout training, allowing the model to increasingly rely on the more reliable and accurate rule-based outcome reward in later stages. In short, our proposed Trust-GRPO enables the model to receive thinking process rewards in a reliable manner, thereby guiding the exploration of favorable and generalizable reasoning strategies.  

In summary, our contributions are as follows:  

. We propose a thinking reward model that evaluates reasoning quality from various dimensions at a holistic level, enabling the model to distinguish between sound and flawed reasoning processes during rule-based RL training.  

: We introduce the Trust-GRPO algorithm, which assigns a trustworthiness weight to thinking rewards based on their reliability. This method guides the model to explore favorable reasoning policies in a trustworthy manner without extra computational overhead. In the experiments, SophiaVL-R1-7B consistently outperforms existing MLLMs on diverse benchmarks (e.g., MathVista, MMMU), highlighting its strong reasoning and generalization abilities. Notably, our SophiaVL-R1-7B outperforms LLaVA-OneVision-72B with $1 0 \times$ more parameters on most benchmarks.  

# 2 Related Work  

# 2.1Process Reward Models  

Reward models (RMs) play a crucial role in guiding and shaping the behavior of models [20, 45]. Several studies [16, 38, 30, 43] demonstrate that process supervision--providing feedback at intermediate reasoning steps-has the potential to enhance reasoning capabilities. For example, [16] introduce powerful Process Reward Models (PRMs) with step-wise rewards, which have been applied to mathematical reasoning [16, 29]. ReST-MCTS\* [41] integrates process supervision and Monte Carlo Tree Search (MCTS) to generate per-step process rewards, enabling efficient selftraining of both policy and reward models without manual annotation. Beyond the text-based domain, VisualPRM [30] extends PRMs to the multimodal domain, achieving significant improvements in the reasoning performance of various MLLMs. Despite these advances, PRMs still face two major challenges: (1) imposing rigid step-wise constraints requires the model to strictly follow predefined reasoning steps, which can limit flexibility and generalization--particularly in general tasks [9, 3]; and (2) evaluating the correctness of individual steps is inherently challenging [45], which may lead models to exploit the reward by repeating valid steps or inserting meaningless ones without making real progress. Therefore, in contrast to prior approaches, we aim to develop a thinking reward model that evaluates reasoning quality from multiple dimensions at a holistic level.  

# 2.2 Multimodal Large Language Model Reasoning  

The field of multimodal large language model reasoning aims to build human-like models capable of handling complex tasks that require understanding and reasoning across multiple modalities [14]. Earlier methods typically depend on fine-grained step-level supervision or learned reward models to guide the reasoning process [35, 30, 40]. In contrast, DeepSeek-R1 [9] demonstrates that reinforcement learning with a rule-based reward model can effectively incentivize strong reasoning abilities without dense supervision. Following the R1 paradigm, several efforts have explored enhancing MLLM reasoning through rule-based reinforcement learning [10, 7, 23, 32, 28]. R1-OneVision [34] introduces a cross-modal reasoning pipeline and adopts a supervised fine-tuning followed by RL strategy to strengthen reasoning capabilities. Curr-ReFT [31] introduces a curriculum-based reinforcement learning paradigm for small-scale MLLMs, combining difficulty-aware rewards and rejection sampling to boost generalization. Video-R1 [7] proposes T-GRPO algorithm to explicitly encourage temporal reasoning in video. Despite their success on multimodal tasks, these approaches rely exclusively on outcome rewards, which often overlook the quality of intermediate reasoning steps.  

# 3 Method  

# 3.1Dataset Composition  

We curate a dataset SophiaVL-R1-130k, comprising 130k examples to support the training of thinking reward model (Section 3.2) and SophiaVL-R1 (Section 3.4). To overcome the scarcity of high-quality multimodal reasoning data and ensure robust model performance across a wide range of tasks, we aggregate samples from a combination of text-only and multimodal datasets, all of which are publicly available. The dataset contains both reasoning-specific tasks and general vision-language understanding tasks. We organize the data into five categories, covering diverse reasoning scenarios, as illustrated in Figure 2 (left).  

![](images/a027eb5f4d970ca08526fdae8aa09199ce089129851198c2a1eca249abc4baf8.jpg)  
Figure 2: Left: Composition of our aggregated dataset SophiaVL-R1-130k from public sources Right: Distribution of the SophiaVL-R1-Thinking- $1 5 6 \mathrm { k }$ dataset used to train the thinking reward model.  

# 3.2 Thinking Reward  

To assess fine-grained reasoning quality of MLLMs' thinking process, we develop a thinking reward model that assigns a score between 0 and 1 based solely on the quality of intermediate reasoning, regardless of whether the final answer is correct.  

To construct the dataset used for training the thinking reward model, we collected 470,331 (question, response) pairs output by Qwen2.5-VL-7B-Instruct [1] during the GRPO training. These data contain both favorable and flawed reasoning patterns occurred in the training. Then, each response is scored by the advanced MLLM, Qwen2.5-VL-72B-Instruct [1], using the prompt in Appendix B. This results in 470,331 (question, response, thinking reward) tuples. The evaluation is based on five dimensions, which are identified from error patterns observed during GRPO training: Logical Soundness, Correct Reasoning, Error Identification, Language Consistency, and Redundancy. Detailed examples of each error pattern are provided in Appendix C.  

To ensure the quality of labels and maintain a balanced distribution across different reward levels. we apply rule-based filtering to remove noisy samples and perform uniform sampling across reward intervals. This process results in 156,703 high-quality annotated samples, with 5,000 to 15,000 samples per interval. Each reward interval corresponds to a discrete range (e.g., [0.0-0.1), [0.1-0.2), .., [0.9-1.0]). The distribution of the full (Annotated) and balanced (Selected) datasets is shown in Figure 2 (right). The resulting dataset is denoted as SophiaVL-R1-Thinking-156k.  

The thinking reward model, initialized with Qwen2.5-VL-3B-Instruct [1], is then trained on this dataset using SFT, where the model is required to output a thinking reward given a question and its corresponding thinking process. Through this training, the thinking reward model learns to identify diverse reasoning errors and assign appropriate rewards accordingly, thereby playing a crucial role in GRPO training by providing feedback on reasoning quality.  

# 3.3Rule-based Outcome Reward  

Following DeepSeek-R1 [9], we construct rule-based outcome reward functions to generate reward signals. Specifically, we design task-specific functions that assess model outputs by comparing them with ground-truth answers. It is worth noting that, to ensure accurate outcome rewards, the majority of training data in SophiaVL-R1-130k are formatted as multiple-choice questions or tasks with numerical answers. Tasks are categorized based on their output formats:  

Numerical: A binary reward is assigned based on an exact match between the predicted and ground-truth values.   
Multiple Choice: The reward is defined based on whether the model's output matches the ground-truth choice.   
: OCR: The reward is computed as the negative Word Error Rate (WER), penalizing transcription inaccuracies.   
Free-form Text: The reward is calculated as the average of ROUGE-1, ROUGE-2, and ROUGE-L scores, measuring n-gram and sequence-level similarity [7].  

![](images/da34da9773e70655a65403b0a10152436a5281f897bf5d0b66ad80e4893c0f65.jpg)  
Figure 3: An illustration of our proposed Trust-GRPO.  

# 3.4Trustworthy Group Relative Policy Optimization (Trust-GRPO)  

As discussed earlier, integrating the thinking reward into GRPO training could help the model distinguish between favorable and flawed reasoning process. Nevertheless, a direct application may result in reward hacking, given that model-generated rewards are not always trustworthy. To deal with this challenge, we introduce the Trust-GRPO algorithm, as illustrated in Figure 3.  

Trust-GRPO optimizes the policy using a combination of two reward types: (1) thinking reward $R ^ { t }$ (Section 3.2) that assigns a score between 0 and 1 based on holistic reasoning quality, and (2) outcome reward $R ^ { o }$ (Section 3.3), derived from rule-based evaluation of outcome answer correctness. To reduce the risk of reward hacking, a trustworthiness weight $\gamma$ is included to determine the influence of thinking reward $R ^ { t }$  

The trustworthiness is computed by contrasting the thinking reward $R ^ { t }$ assigned to responses that arrive at correct answers with those leading to incorrect ones. When higher thinking rewards are abnormally associated with incorrect reasoning, $\gamma$ will be lower, indicating the potential unreliability in the reward signal. Next, we will introduce how to derive it.  

First, responses $o _ { i }$ to a question $q$ are grouped into correct answer group $G _ { \mathrm { c o r r e c t } }$ and wrong answer group $G _ { \mathrm { w r o n g } }$ based on their outcome rewards. Then, we calculate the average thinking reward in $G _ { \mathrm { c o r r e c t } }$ and $G _ { \mathrm { w r o n g } }$ as follows:  

$$
\begin{array} { l } { \displaystyle \mu _ { c } = \frac { 1 } { \left| G _ { \mathrm { c o r r e c t } } \right| } \sum _ { i \in G _ { \mathrm { c o r r e c t } } } R _ { i } ^ { t } , \quad \displaystyle G _ { \mathrm { c o r r e c t } } = \left\{ i \mid R _ { i } ^ { o } \geq 0 . 5 \right\} , } \\ { \displaystyle \mu _ { w } = \frac { 1 } { \left| G _ { \mathrm { w r o n g } } \right| } \sum _ { i \in G _ { \mathrm { w r o n g } } } R _ { i } ^ { t } , \quad \displaystyle G _ { \mathrm { w r o n g } } = \left\{ i \mid R _ { i } ^ { o } < 0 . 5 \right\} , } \end{array}
$$  

where $\mu _ { c }$ and $\mu _ { w }$ denote the average outcome rewards in the correct answer group and the wrong answer group, respectively. $R _ { i } ^ { o }$ denotes the outcome reward of response $i$ . The trustworthiness weight $\gamma$ is defined as follows:  

$$
\gamma = \left\{ \begin{array} { l l } { 1 , } & { \mu _ { c } \geq \mu _ { w } } \\ { e ^ { \mu _ { c } - \mu _ { w } } , } & { \mu _ { c } < \mu _ { w } } \end{array} . \right.
$$  

![](images/ed0e97ce17913a46d4c933d922376d96d9bd4ac647aa1a016be889e273575d17.jpg)  
Figure 4: Example of trustworthiness weight $\gamma$ . Incorrect responses (red) receive higher average thinking rewards than correct ones (green), indicating misalignment between $R ^ { t }$ and $R ^ { o }$ and the need for a trustworthiness-aware adjustment.  

This comparison between $\mu _ { c }$ and $\mu _ { w }$ allows us to assess the alignment between thinking rewards and rule-based outcome rewards. A lower $\gamma$ indicates a discrepancy between $R ^ { t }$ and $R ^ { o }$ , suggesting that the thinking reward may be unreliable for this response group and thus should be given reduced weight. $\gamma$ dynamically estimates the trustworthiness of thinking rewards for each question's response group without incurring additional computational overhead.  

The $i$ -th reward $R _ { i }$ incorporating the thinking reward with trustworthiness weight is defined as:  

$$
R _ { i } = R _ { i } ^ { o } + \gamma \alpha \cdot R _ { i } ^ { t } ,
$$  

where $\alpha$ is a hyperparameter that controls the impact of thinking reward.  

We further introduce a time-based annealing strategy that gradually reduces the influence of thinking reward as training progresses. This encourages the reasoning model to rely increasingly on the more accurate rule-based outcome reward in later steps. Combining both components, the final reward $R _ { i }$ is defined as:  

$$
R _ { i } = R _ { i } ^ { o } + \gamma \alpha e ^ { - \frac { \mathrm { s t e p s } } { T } } \cdot R _ { i } ^ { t } ,
$$  

where steps denotes the current global training step and $T$ is the total number of training steps, controlling the decay rate of thinking reward's influence over time.  

The advantage $A _ { i }$ is computed using rewards of each response group:  

$$
A _ { i } = { \frac { R _ { i } - \operatorname * { m e a n } ( \{ R _ { 1 } , R _ { 2 } , \cdots , R _ { N } \} ) } { \operatorname * { s t d } ( \{ R _ { 1 } , R _ { 2 } , \cdots , R _ { N } \} ) } } ,
$$  

Following DeepSeek-R1 [9], given a question $q$ GRPO samples a group of responses $o _ { 1 } , o _ { 2 } , \ldots , o _ { N }$ from the old policy $\pi _ { \mathrm { o l d } }$ , and updates the policy $\pi _ { \boldsymbol { \theta } }$ by maximizing the following objective:  

$$
\begin{array} { r l } & { \mathcal { I } _ { G R P O } ( \theta ) = \mathbb { E } \left[ q \sim P ( Q ) , \ \{ \sigma _ { i } \} _ { i = 1 } ^ { N } \sim \pi _ { \mathrm { o l d } } ( O | q ) \right] } \\ & { \quad \frac { 1 } { N } \displaystyle \sum _ { t = 1 } ^ { N } \left( \operatorname* { m i n } \left( \frac { \pi _ { \theta } \left( \sigma _ { i } | q \right) } { \pi _ { \mathrm { o l d } } \left( \sigma _ { i } | q \right) } A _ { i } , \ \mathrm { c l i p } \left( \frac { \pi _ { \theta } \left( \sigma _ { i } | q \right) } { \pi _ { \mathrm { o l d } } \left( \sigma _ { i } | q \right) } , 1 - \epsilon , 1 + \epsilon \right) A _ { i } \right) - \beta \mathbb { D } _ { \mathsf { K L } } [ \pi _ { \theta } | | \pi _ { \mathrm { r e f } } | \right) . } \end{array}
$$  

By contrasting the thinking rewards of correct and incorrect responses, Trust-GRPO improves the reliability of reward signals, thereby encouraging more generalizable reasoning behavior.  

Figure 4 illustrates a case where the trustworthiness weight $\gamma$ helps identify potentially unreliable thinking rewards. Responses with incorrect answers are shown in red and those with correct answers  

Table 1: Comparison of models on MathVista and MathVerse. The best is bold, and the runner-up is underline. 'Scientific Reasoning, 2Textbook Question Answering, Arithmetic Reasoning, 4Math Word Problem, 'Logical Reasoning, Vision Intensive, 'Vision Only, 'Vision Dominant, Text Dominant, 1Text Lite.   


<html><body><table><tr><td rowspan="2">Model</td><td colspan="6">MathVista</td><td colspan="6">MathVerse</td></tr><tr><td>AVG</td><td>SCI!</td><td>TQA?</td><td>ARI3</td><td>MWp4</td><td>LOG5</td><td>AVG</td><td>VI6</td><td>vo7</td><td>VD8</td><td>TD9</td><td>TL 10</td></tr><tr><td>Open-Source General MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-OneVision-7B [12]</td><td>63.2</td><td>65.6</td><td>60.8</td><td>57.8</td><td>69.4</td><td>21.6</td><td>26.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-OneVision-72B [12]</td><td>68.4</td><td>63.1</td><td>65.8</td><td>60.1</td><td>73.7</td><td>27.1</td><td>27.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cambrian-1-34B [27]</td><td>50.9</td><td>53.3</td><td>55.1</td><td>45.6</td><td>51.6</td><td>16.2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4V</td><td>51.8</td><td>63.1</td><td>65.8</td><td>51.8</td><td>57.5</td><td>21.6</td><td>32.8</td><td>-</td><td></td><td></td><td></td><td></td></tr><tr><td>Open-Source Math MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Math-LLaVA-13B [25]</td><td>46.6</td><td>49.2</td><td>51.3</td><td>40.2</td><td>56.5</td><td>16.2</td><td>22.9</td><td>24.5</td><td>16.1</td><td>21.7</td><td>27.3</td><td>24.9</td></tr><tr><td>Math-PUMA-Qwen2VL-7B [46]</td><td>47.9</td><td>42.6</td><td>46.2</td><td>46.2</td><td>68.3</td><td>21.6</td><td>33.6</td><td>33.4</td><td>26.0</td><td>31.6</td><td>42.1</td><td>35.0</td></tr><tr><td>Multimath-7B [21]</td><td>50.0</td><td></td><td>50.0</td><td></td><td>61.8</td><td></td><td>26.9</td><td>28.1</td><td>15.0</td><td>25.9</td><td>34.8</td><td>30.8</td></tr><tr><td>URSA-8B [18]</td><td>59.8</td><td>58.2</td><td>63.9</td><td>53.5</td><td>75.3</td><td>21.6</td><td>45.7</td><td>46.4</td><td>34.6</td><td>43.9</td><td>55.3</td><td>48.3</td></tr><tr><td>Open-Source Reasoning MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Curr-ReFT-7B [4]</td><td>64.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>R1-OneVision-7B [34]</td><td>64.1</td><td>61.5</td><td>62.0</td><td>56.1</td><td>64.5</td><td>16.2</td><td>46.4</td><td></td><td>40.0</td><td></td><td></td><td></td></tr><tr><td>InternVL2.5-8B-VisualPRM [30]</td><td>68.5</td><td>61.5</td><td>53.9</td><td>45.9</td><td>66.8</td><td>21.2</td><td>30.7</td><td>28.9</td><td>35.8</td><td>27.3</td><td>31.7</td><td>29.7</td></tr><tr><td>Qwen2.5-VL-7B-Instruct [1]</td><td>67.5</td><td>65.6</td><td>67.7</td><td>57.5</td><td>69.4</td><td>27.0</td><td>44.0</td><td>41.1</td><td>41.0</td><td>38.7</td><td>55.2</td><td>44.0</td></tr><tr><td>+GRPO</td><td>69.9</td><td>68.0</td><td>69.6</td><td>61.2</td><td>75.8</td><td>24.3</td><td>45.3</td><td>43.0</td><td>41.0</td><td>41.1</td><td>56.0</td><td>45.6</td></tr><tr><td>+SFT+GRPO</td><td>66.8</td><td>72.1</td><td>73.4</td><td>59.8</td><td>69.9</td><td>21.6</td><td>43.1</td><td>42.5</td><td>37.1</td><td>37.3</td><td>52.2</td><td>46.3</td></tr><tr><td>SophiaVL-R1-7B</td><td>71.3</td><td>70.5</td><td>73.4</td><td>62.6</td><td>76.9</td><td>35.1</td><td>48.8</td><td>45.4</td><td>43.9</td><td>45.1</td><td>58.5</td><td>51.3</td></tr></table></body></html>  

in green. Despite being incorrect, the red group receives a higher average thinking reward, indicating a misalignment between $R ^ { t }$ and $R ^ { o }$ . This implies that the thinking reward has potential risk of unreliability, thus should be assigned less weight. More examples can be found in Appendix D.  

# 4 Experiment  

# 4.1 Experiment Settings  

Benchmarks. We evaluate our model on both multimodal mathematical reasoning and general multimodal reasoning benchmarks. For mathematical reasoning, we report detailed results on MathVista [17] and MathVerse [42]. For general multimodal capabilities, we conduct evaluations on MMMU [39], MME [15], MMStar [2], ChartQA [19], and MMBench [33].  

Implementation Details.The thinking reward model is initialized from Qwen2.5-VL-3B-Instruct and trained using SFT for 2 epochs on 4 NVIDIA A800 80GB GPUs on SophiaVL-R1-Thinking156k dataset. We initialize the reasoning model with Qwen2.5-VL-7B-Instruct and train it on our SophiaVL-R1-130k dataset using the proposed Trust-GRPO algorithm. The RL training is conducted for 1,500 steps with VeRL [44, 24] on 8 NVIDIA A800 80GB GPUs. The group size is set to 8, the KL divergence coefficient to 0.04, and the learning rate to $5 \times 1 0 ^ { - 7 }$ . The hyperparameter $\alpha$ is set to 0.3. During evaluation, we use default prompts and apply greedy decoding to generate responses. Additional evaluation details are provided in Appendix A.  

# 4.2 Main Results  

Performance on Math Reasoning Benchmarks. As shown in Table 1, SophiaVL-R1-7B achieves competitive performance on mathematical reasoning benchmarks. On the MathVista benchmark, it attains an accuracy of $7 1 . 3 \%$ , surpassing both Qwen2.5-VL-7B-Instruct models trained with GRPO and $_ { \mathrm { S F T + G R P O } }$ strategies, and also outperforming the LLaVA-OneVision-72B model. Compared to the model trained by VisualPRM [30], our model achieves significantly better performance, with an 18.2-point improvement on MathVerse (48.9 vs. 30.7), and consistently outperforms it across all sub-tasks. These results indicate that, compared to PRM-based method, our Trust-GRPO may serve as a more effective approach for providing reward signals, beter guiding the model toward improved reasoning ability.  

Table 2: Comparison of models on general ability benchmarks. The best is bold, and the runner-up is underline.   


<html><body><table><tr><td>Model</td><td>MMMU</td><td>MME</td><td>ChartQA</td><td>MMBench</td><td>MMStar</td></tr><tr><td>Open-Source General MLLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-OneVision-7B [12]</td><td>48.8</td><td>1998.0</td><td>80.0</td><td></td><td>61.7</td></tr><tr><td>LLaVA-OneVision-72B [12]</td><td>56.8</td><td>2261.0</td><td>83.7</td><td></td><td>66.1</td></tr><tr><td>Cambrian-1-34B [27]</td><td>49.7</td><td>1689.3</td><td>75.6</td><td>81.4</td><td>54.2</td></tr><tr><td>GPT-4V</td><td>56.8</td><td>1926.0</td><td>78.5</td><td>75.0</td><td>57.1</td></tr><tr><td>Open-Source Math MLLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>URSA-8B [18]</td><td>43.1</td><td>1605.7</td><td>44.4</td><td>55.5</td><td>42.3</td></tr><tr><td>Open-Source Reasoning MLLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Curr-ReFT-7B [4]</td><td></td><td></td><td></td><td>79.0</td><td></td></tr><tr><td>R1-Onevision-7B [34]</td><td>51.6</td><td>2223.3</td><td></td><td>75.6</td><td>59.1</td></tr><tr><td>InternVL2.5-8B-VisualPRM [30]</td><td>56.2</td><td></td><td>60.8</td><td>83.5</td><td>63.4</td></tr><tr><td>Qwen2.5-VL-7B-Instruct [1]</td><td>57.4</td><td>2306.0</td><td>86.3</td><td>83.3</td><td>64.3</td></tr><tr><td>+GRPO</td><td>58.0</td><td>2298.2</td><td>87.2</td><td>83.4</td><td>65.6</td></tr><tr><td>+SFT+GRPO</td><td>59.1</td><td>2344.1</td><td>89.2</td><td>84.6</td><td>64.7</td></tr><tr><td>SophiaVL-R1-7B</td><td>61.3</td><td>2403.8</td><td>88.5</td><td>85.4</td><td>66.7</td></tr></table></body></html>  

Performance on General Benchmarks. Many task-specific reasoning models, such as those optimized for mathematical problem-solving or other specialized tasks, excel within their respective domains but often struggle to maintain strong performance on general multimodal benchmarks (e.g., URSA-8B). Different from them, SophiaVL-R1-7B demonstrates consistently strong performance across widely recognized general ability benchmarks, as shown in Table 2, highlighting its superior generalization capability. For example, on the widely used MMMU benchmark for multi-discipline reasoning, SophiaVL-R1-7B outperforms LLaVA-OneVision-72B by 4.5 points.  

# 5 Ablation Study  

We conduct ablation studies to examine the contributions of key components in our method. Specifically, we evaluate three variants of our SophiaVL-R1:  

SophiaVL-R1-wo-trained-TRM: replacing the trained thinking reward model with an untrained Qwen2.5-VL-3B-Instruct model.   
SophiaVL-R1-wo-trust-and-annealing: removing both the trustworthiness weighting and the annealing strategy from Trust-GRPO.   
SophiaVL-R1-wo-trust: removing only the trustworthiness weight while retaining the time-based annealing schedule.  

Besides, we also include Qwen2.5-VL-7B $^ +$ GRPO as a baseline, which directly uses GRPO for training Qwen2.5-VL-7B-Instruct. The results are summarized in Table 3.  

Effect of the Thinking Reward Model. It can be found that SophiaVL-R1-wo-trained-TRM consistently obtains worse performance than SophiaVL-R1. This highlights the effectiveness of our training pipeline and the SophiaVL-R1-Thinking-156k dataset in improving the thinking reward model's ability to provide accurate and informative reward signals for reasoning optimization. What's more, replacing the thinking reward model with an untrained version still yields improvements over the Qwen2.5-VL-7B $+$ GRPO baseline. This suggests that incorporating holistic-level thinking rewards contributes to more effective reasoning model training, even without reward model training.  

Effect of the Trustworthiness Weight $\gamma$ .We observe a performance drop across all benchmarks in SophiaVL-R1-wo-trust when the trustworthiness weight is removed, compared to the full SophiaVL.  

Table 3: Ablation Study.   


<html><body><table><tr><td>Model</td><td>MathVistaMathVerse</td><td></td><td>MMMU</td><td>MME</td><td>ChartQA</td><td>MMBench</td><td>MMStar</td></tr><tr><td>Qwen2.5-VL-7B+GRPO</td><td>67.5</td><td>44.0</td><td>57.4</td><td>2306.0</td><td>86.3</td><td>83.3</td><td>64.4</td></tr><tr><td>SophiaVL-R1-wo-trained-TRM</td><td>68.4</td><td>47.9</td><td>57.0</td><td>2347.1</td><td>87.7</td><td>84.0</td><td>65.7</td></tr><tr><td>SophiaVL-R1-wo-trust-and-annealing 67.4</td><td></td><td>46.3</td><td>56.7</td><td>2366.8</td><td>86.3</td><td>82.6</td><td>65.0</td></tr><tr><td>SophiaVL-R1-wo-trust</td><td>70.2</td><td>47.8</td><td>60.0</td><td>2363.3</td><td>87.8</td><td>83.7</td><td>65.2</td></tr><tr><td>SophiaVL-R1</td><td>71.3</td><td>48.9</td><td>61.3</td><td>2403.8</td><td>88.5</td><td>84.5</td><td>66.7</td></tr></table></body></html>  

![](images/cd74c8795f7d48b240bbece16f5e53e35637c0cbd35de6ecc64ae49028b5fa4e.jpg)  
Figure 5: Training curves of mean rule-based outcome reward across different methods.  

R1 model. This demonstrates the effectiveness of trustworthiness weighting, which allows the model to receive thinking process rewards in a more reliable manner.  

Effect of the Time-based Annealing Strategy. To assess the effect of time-based annealing. we compare SophiaVL-R1-wo-trust-and-annealing with SophiaVL-R1-wo-trust. We can find that SophiaVL-R1-wo-trust-and-annealing generally performs worse than SophiaVL-R1-wo-trust on most benchmarks. The performance drop may be due to the over-exploitation of the thinking reward, where potentially unreliable signals could interfere with the optimization of the reasoning policy. This suggests that gradually reducing the influence of the thinking reward by our proposed annealing strategy is beneficial, as it encourages reliance on the more reliable rule-based outcome reward in later training stages.  

Training Curve Analysis. Figure 5 shows the average outcome reward per training step for each method. Compared to all baselines and ablated variants, SophiaVL-R1 achieves the highest reward and demonstrates a faster improvement throughout training. While some variants achieve moderate reward levels, SophiaVL-R1-wo-trust-and-annealing exhibits clearly unstable learning dynamics Overall, these results highlight the importance of both trustworthiness weighting and time-based annealing in ensuring stable and effective training.  

# 6 Conclusion  

In this work, we propose SophiaVL-R1, a multimodal large language model trained using a novel Trust-GRPO algorithm that integrates model-generated thinking rewards with rule-based outcome rewards. To promote generalizable reasoning, we introduce a holistic-level thinking reward model that assesses the quality of reasoning processes. Furthermore, we mitigate the challenge of reward hacking by introducing a trustworthiness weighting mechanism together with a time-based annealing strategy. Experimental results across multiple MLLM benchmarks demonstrate that SophiaVL-R1 consistently outperforms existing MLLMs, even outperforming $1 0 \times$ larger model. Our findings highlight the value of thinking process supervision beyond final correctness and offer insights for future studies on developing reasoning models.  

References   
[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
[2] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.   
[3] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.   
[4] Huilin Deng, Ding Zou, Rui Ma, Hongchen Luo, Yang Cao, and Yu Kang. Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning. arXiv preprint arXiv:2503.07065, 2025.   
[5] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative selfimprovement. arXiv preprint arXiv:2503.17352, 2025.   
[6] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198-11201, 2024.   
[7] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.   
[8] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050-1059. PMLR, 2016.   
[9] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in 1lms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[10] Yuxiang Lai, Jike Zhong, Ming Li, Shitian Zhao, and Xiaofeng Yang. Med-r1: Reinforcement learning for generalizable medical reasoning in vision-language models. arXiv preprint arXiv:2503.13939, 2025.   
[11] Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Hang Zhang, Yuming Jiang, Xin Li, Deli Zhao, Fan Wang, Yu Rong, Aixin Sun, and Shijian Lu. Mmr1: Advancing the frontiers of multimodal reasoning. https://github. com/LengSicong/MMR1, 2025.   
[12] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.   
[13] Dawei Li, Renliang Sun, Yue Huang, Ming Zhong, Bohan Jiang, Jiawei Han, Xiangliang Zhang, Wei Wang, and Huan Liu. Preference leakage: A contamination problem in llm-as-a-judge. arXiv preprint arXiv:2502.01534, 2025.   
[14] Yunxin Li, Zhenyu Liu, Zitao Li, Xuanyu Zhang, Zhenran Xu, Xinyu Chen, Haoyuan Shi, Shenyuan Jiang, Xintong Wang, Jifang Wang, et al. Perception, reason, think, and plan: A survey on large multimodal reasoning models. arXiv preprint arXiv:2505.04921, 2025.   
[15] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. A survey of multimodel large language models. In Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering, pages 405-409, 2024.   
[16] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In The Twelfth International Conference on Learning Representations, 2023.   
[17] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.   
[18] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Yiyao Yu, Xinzhe Ni, Zicheng Lin, Jin Zeng, and Yujiu Yang. Ursa: Understanding and verifying chain-of-thought reasoning in multimodal mathematics. arXiv preprint arXiv:2501.04686, 2025.   
[19] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.   
[20] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730-27744, 2022.   
[21] Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147, 2024.   
[22] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
[23] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.   
[24] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.   
[25] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294, 2024.   
[26] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. Advances in Neural Information Processing Systems, 35:9460- 9471, 2022.   
[27] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310-87356, 2024.   
[28] Junke Wang, Zhi Tian, Xun Wang, Xinyu Zhang, Weilin Huang, Zuxuan Wu, and Yu-Gang Jiang. Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl. arXiv preprint arXiv:2504.11455, 2025.   
[29] Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. arXiv preprint arXiv:2312.08935, 2023.   
[30] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, et al. Visualprm: An effective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291, 2025.   
[31] Jinyang Wu, Mingkuan Feng, Shuai Zhang, Ruihan Jin, Feihu Che, Zengqi Wen, and Jianhua Tao. Boosting multimodal reasoning with mcts-automated structured thinking. arXiv preprint arXiv:2502.02339, 2025.   
[32] Xiaobo Xia and Run Luo. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.   
[33] Cheng Xu, Xiaofeng Hou, Jiacheng Liu, Chao Li, Tianhao Huang, Xiaozhi Zhu, Mo Niu, Lingyu Sun, Peng Tang, Tongqiao Xu, et al. Mmbench: Benchmarking end-to-end multi-modal dnns and understanding their hardware-software implications. In 2023 IEEE International Symposium on Workload Characterization (IISwC), pages 154-166. IEEE, 2023.   
[34] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.   
[35] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319, 2024.   
[36] Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, et al. Justice or prejudice? quantifying biases in 1lm-as-a-judge. arXiv preprint arXiv:2410.02736, 2024.   
[37] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.   
[38] Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, Kaiyan Zhang, Bowen Zhou, Zhiyuan Liu, and Hao Peng. Free process rewards without process labels. arXiv preprint arXiv:2412.01981, 2024.   
[39] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556-9567, 2024.   
[40] Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Ziyu Liu, Shengyuan Ding, Shenxi Wu, Yubo Ma, Haodong Duan, Wenwei Zhang, et al. Internlm-xcomposer2. 5-reward: A simple yet effective multi-modal reward model. arXiv preprint arXiv:2501.12368, 2025.   
[41] Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts\*: Llm self-training via process reward guided tree search. Advances in Neural Information Processing Systems, 37:64735-64772, 2024.   
[42] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169-186. Springer, 2024.   
[43] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301, 2025.   
[44] Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework. https : //github. com/hiyouga/EasyR1, 2025.   
[45] Jialun Zhong, Wei Shen, Yanzeng Li, Songyang Gao, Hua Lu, Yicheng Chen, Yang Zhang, Wei Zhou, Jinjie Gu, and Lei Zou. A comprehensive survey of reward models: Taxonomy, applications, challenges, and future. arXiv preprint arXiv:2504.12328, 2025.  

[46] Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. Math-puma: Progressive up. ward multimodal alignment to enhance mathematical reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 26183-26191, 2025.  

# A Evaluation Details  

Most of our evaluations are conducted using VLMEvalKit [6], following the recommended Python package versions. For baseline models, performance metrics are obtained from the OpenVLM leaderboard. We adopt the default prompts for all evaluated models and modify the answer extraction function based on each model's output format. For instance, for R1-style models, we extract the content enclosed within the <answer> and </answer> tags.  

For MathVista, we evaluate on the testmini split. For MathVerse, we report average performance over the following subsets: vision-only, vision-dominant, vision-intensive, text-dominant, and textlite. For MMMU, we evaluate on the mmmu_dev_val set. For ChartQA, evaluation is conducted on the test set. For MMBench, we use the MMBench_Dev_EN set for evaluation.  

# B Prompt Used for Evaluating Thinking Process Quality  

Table 4: Prompt for evaluating thinking process quality.   


<html><body><table><tr><td colspan="2">Input {Image}, {Question} and {Model Response}</td></tr><tr><td colspan="2">You are an expert reasoning evaluator. I will give you a multimodal question and an answer. Your goal is to judge a reward process and give a score between 0 and 1. You should focus on whether</td></tr><tr><td colspan="2">the reasoning process is good rather than whether the final answer is correct. Evaluation Criteria:</td></tr><tr><td>1. Logical Soundness 2. Correct Reasoning</td><td>Does each step follow logically from the previous one? Are the methods and steps used appropriate and valid? Are the facts</td></tr><tr><td>3. Error Identification</td><td>and lemmas correctly stated and applied?. Are there logical flaws, unsupported assumptions, or incorrect steps?</td></tr><tr><td>4. Language Consistency</td><td>Is the reasoning process conducted in a single, consistent language without mixing different languages?.</td></tr><tr><td colspan="2">5. Redundancy Is the reasoning concise, avoiding repetition or irrelevant steps? Provide a single score from {0, 0.1, 0.2, ..., 1.0} based on the reasoning quality, where:</td></tr></table></body></html>

Be strict, reward the good process and punish the bad one. You should only output the score without any explanation.  

# CError Patterns Observed in GRPO Training  

We present additional examples illustrating error paterns we observed in the reasoning process during GRPO training. Figure 6 highlights two typical issues: the Logical Soundness and Error Identification patterns. In this case, the model incorrectly extracts relationships between angles and performs faulty equation calculations. Figure 7 also exemplifies the Error Identification pattern, where the model misinterprets visual conditions by confusing edge length information with angle values, ultimately leading to incorrect reasoning.  

![](images/fa9d9aa998fd89d0a269dcc5e4a9cf486e90fa4cb4d8b2113ff636d4e0ddc7a2.jpg)  
Figure 6: Examples of wrong thinking and performance of thinking reward model.  

![](images/823fef2d88321820d90788849d35a7731f490f83a50e3c14f1ead64af27de5a4.jpg)  
Figure 7: Examples of wrong thinking and performance of thinking reward model.  

# D Case Study of $\gamma$  

We demonstrate a text-only mathematical problem case in Figure 8. All responses in this image corresponded to the same question displayed on the top. The ground truth answer is 14. Responses yielding incorrect answers (e.g., RESPONSE[6]) are highlighted in red (grouped as $G _ { w r o n g } \$ , while while those producing correct answers (e.g., RESPONSE[8]) are highlighted in green $\mathbf { \bar { \Pi } } _ { G _ { c o r r e c t } }$ Notably, RESPONSE[6] receives a thinking reward of 0.7 despite a clear arithmetic simplification error, exceeding the reward assigned to RESPONSE[8]. By computing the average thinking reward of $G _ { c o r r e c t }$ and $G _ { w r o n g }$ . we obtain a trustworthiness weight of $\gamma = 0 . 7 4$ . Since $\gamma < 1$ , this indicates the presence of potential unreliable thinking rewards within this response group. This case demonstrates how our Trust-GRPO algorithm can adaptively identify such unreliability during training and appropriately downscale its influence by adjusting $\gamma$ thereby providing more stable and reliable reward signals for effective GRPO training.  

![](images/c25096ea7f7ed1ba06b148f693940b22c11dbcf69e50aa9ca4c2614268195f2e.jpg)  
Figure 8: Case of a text-only mathematical problem.  

We present a free-form reasoning example in Figure 9, with the ground-truth answer shown at the top. Notably, RESPONSE[6] identifies six key points, while RESPONSE[8] correctly identifies eight--demonstrating a more complete reasoning process. However, RESPONSE[8] receives a thinking reward of only 0.0, significantly lower than the 0.8 assigned to RESPONSE[6]. Together with the first case study and the analysis provided in the main paper, this example illustrates how the trustworthiness weight $\gamma$ functions as an effective dynamic parameter for identifying unreliable thinking rewards. By adaptively down-weighting suspicious reward signals, $\gamma$ helps mitigate the risk of reward hacking and contributes to more stable reinforcement learning.  

![](images/529a7946d29523e1b5c6a699f95459d2c9f6a926ba854b630c06b3f57248217d.jpg)  
Figure 9: Case of a free-form problem.  

# E Reasoning Examples of SophiaVL-R1-7B  

![](images/fdaa7fa5837d73f9ed04771dea22a9cefe78f7615dc448408a3689c39f2f0e68.jpg)  
Figure 10: An example of SophiaVL-R1-7B's reasoning output on MathVista.  

![](images/a6e8574843b30d05614ee3d6564d809fbf9a5e40b7390632cc40d23a18923505.jpg)  
Figure 11: An example of SophiaVL-R1-7B's reasoning output on MME.  

Which image was generated by AI on <image 1>?  

A. The left one B. The right one C. Both images D. Neither image  

![](images/19bed9d3ddb5fd31b43e151a84383057fbc2e46cd2eda35611617ffebf9eba70.jpg)  

<think>Let me think about this. The images appear to be paintings, which are not typically generated by AI. However, the question seems to be asking which one was created by AI, which is unusual for a question about art. Given the options, it's likely that the question is asking which one is not a traditional painting, which would be the AI-generated image. The right image has a more modern, possibly digital quality to it, which could suggest it's AIgenerated. The left image looks more like a traditional painting with a classic style.   
Hmm, I see. The right image has a more contemporary look, which might suggest it's AI-generated. The left image looks more like a traditional painting.   
Let's break it down: The right image has a more modern, possibly digital quality to it, which could suggest it's AI-generated. The left image looks more like a traditional painting.   
Therefore, the right image is the one that was likely generated by AI</think>   
<answer>B. The right one.</answer>  

What feeling is represented in this image?  

A. engaged B. disordered C. angry D. supportive <think>Let me think about this image.  

![](images/7ac9c8fa50f00a562d0aa1a6a86d7d5f4674bb99e2b6064e5e97388e6f3290e2.jpg)  
Figure 12: An example of SophiaVL-R1-7B's reasoning output on MMMU.   
Figure 13: An example of SophiaVL-R1-7B's reasoning output on MMStar.  

The person in the picture appears to be in a state of confusion or distress, as indicated by the chaotic lines around their head, which resemble a brain with tangled thoughts. The leaves floating around might symbolize a sense of being overwhelmed or scattered. The person's hand on their head suggests they are trying to make sense of something.  

Now, let's look at the options:  

A. engaged - this doesn't seem to fit as the person doesn't appear to be focused on something.   
B. B. disordered - this aligns well with the chaotic lines around the head.   
C. C. angry - there's no clear indication of anger, just confusion.   
D. D. supportive - the image doesn't show any signs of support or encouragement.  

Therefore, the most fitting option is B. disordered.</think> <answer>B. disordered</answer>  

# F Limitations and Future Works  

We discuss the limitations of our work and potential directions for future research as follows:  

First, our work employs a holistic thinking reward model that directly outputs scalar rewards without explicitly performing reasoning before scoring. Future work could explore enabling the thinking reward model to generate reasoning trajectories before assigning a reward, in order to improve the reliability and stability of the reward signals. Second, while this work focuses on image and text modalities, future research could extend the proposed approach to other modalities, including video and audio.  