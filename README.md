# RoutingModel – Customer Query Routing with LLM Fine-Tuning

## 📌 Overview

**RoutingModel** is a fine-tuned Large Language Model designed to automatically route customer queries to the appropriate department or support team.

The model is fine-tuned using **Llama 3.2 3B Instruct** with **Unsloth** and **LoRA/PEFT**. Given a customer query, the model analyzes the intent and returns **only the corresponding department label**, making it suitable for automated customer-service routing systems.

### Example

**Input:**

```text
I never received the account confirmation email for our company.
```

**Output:**

```text
SUPPORT_TEAM
```

Another example from the training dataset:

**Input:**

```text
I need to add another administrator to our workspace.
```

**Output:**

```text
ACCOUNT_TEAM
```

## 🚀 Features

* 🤖 LLM-based customer query classification
* 🎯 Automatic routing to the correct department
* ⚡ Efficient fine-tuning using Unsloth
* 🧠 Llama 3.2 3B Instruct as the base model
* 🔧 LoRA/PEFT parameter-efficient fine-tuning
* 💾 4-bit quantization for memory-efficient training
* 📝 Instruction-based training using chat templates
* 🔤 Generates a single department label as the final prediction

## 🏗️ Model Architecture

**Base Model:** `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`

**Fine-Tuning Method:** LoRA / PEFT

**Quantization:** 4-bit

**Maximum Sequence Length:** 2048

The model uses LoRA adapters on the attention and MLP projection layers, including:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The LoRA configuration uses:

```text
r = 16
lora_alpha = 16
lora_dropout = 0
```

## 📊 Dataset

The model was trained using a **JSONL customer-service classification dataset** containing input queries and their corresponding department labels.

Dataset format:

```json
{"input": "I never received the account confirmation email for our company.", "output": "SUPPORT_TEAM"}
{"input": "I need to add another administrator to our workspace.", "output": "ACCOUNT_TEAM"}
```

The dataset was loaded using the Hugging Face `datasets` library and converted into instruction-style conversations.

## 🧩 Prompt Format

The model was trained with the following routing instruction:

```text
You are an intelligent routing assistant.
Classify the query into the correct department.
Output only the label (Eg: SUPPORT_TEAM)
```

Each training example follows this structure:

```text
System:
You are an intelligent routing assistant. Classify the query into the correct department.

User:
<Customer Query>

Assistant:
<Department Label>
```

## ⚙️ Training

The model was fine-tuned using **TRL's SFTTrainer** together with Unsloth.

Key training configuration:

```text
Batch Size              : 2
Gradient Accumulation   : 4
Learning Rate           : 2e-4
Maximum Steps           : 60
Optimizer               : AdamW 8-bit
Weight Decay            : 0.01
Scheduler               : Linear
Warmup Steps            : 5
Sequence Length         : 2048
```

Unsloth was used to optimize the fine-tuning process and reduce GPU memory requirements.

## 🔄 Workflow

```text
Customer Query
      ↓
Tokenizer
      ↓
Fine-Tuned Llama 3.2 3B
      ↓
Intent Classification
      ↓
Department Label
      ↓
Customer Query Routing
```

### Example

```text
Customer:
"I have a problem with my account confirmation."

             ↓

RoutingModel

             ↓

SUPPORT_TEAM
```

## 💡 Use Cases

This model can be integrated into:

* Customer support automation
* Helpdesk ticket routing
* Email classification
* Chatbot systems
* Customer-service workflows
* CRM automation
* Multi-agent AI systems
* Automated ticket assignment

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Llama 3.2 3B Instruct
* Unsloth
* LoRA / PEFT
* TRL
* Hugging Face Datasets
* CUDA
* 4-bit Quantization

## 🎯 Project Goal

The goal of **RoutingModel** is to create a lightweight and efficient LLM-based routing system that can understand customer queries and automatically direct them to the appropriate team without generating unnecessary responses.

Instead of producing a complete conversational answer, the model focuses on one specific task:

```text
Customer Query → Correct Department
```

This makes it suitable as a **routing component inside larger AI customer-service systems**.
