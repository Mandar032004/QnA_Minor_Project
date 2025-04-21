import json
import torch
import pandas as pd
import re
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import evaluate

def safe_eval(val):
    try:
        match = re.search(r"\['(.*?)'\]", val.replace('\n', ' '))
        if match:
            answers = match.group(1).split("', '")
            return {"text": answers}
        else:
            return {"text": []}
    except Exception as e:
        print(f"Error parsing: {val} - {e}")
        return {"text": []}

def load_squad(train_file, val_file):
    with open(train_file, 'r', encoding='utf-8') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []

    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for ans in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(ans)

    train_df = pd.DataFrame({
        'context': contexts,
        'question': questions,
        'answers': answers
    })

    val_df = pd.read_csv(val_file)
    val_df["answers"] = val_df["answers"].apply(safe_eval)

    return train_df, val_df

def add_token_positions(df):
    start_positions = []
    for i in range(len(df)):
        context = df.loc[i, 'context']
        ans = df.loc[i, 'answers']['text'][0] if df.loc[i, 'answers']['text'] else ''
        start_pos = context.find(ans)
        if start_pos == -1:
            start_pos = 0
        start_positions.append({"text": [ans], "answer_start": [start_pos]})
    df['answers'] = start_positions
    return df

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def preprocess(example):
    inputs = tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )
    start_char = example["answers"]["answer_start"][0]
    end_char = start_char + len(example["answers"]["text"][0])
    offsets = inputs.pop("offset_mapping")

    start_token = end_token = 0
    for i, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_token = i
        if start < end_char <= end:
            end_token = i
            break

    inputs["start_positions"] = start_token
    inputs["end_positions"] = end_token
    return inputs

def plot_loss(log_history):
    train_loss = [log["loss"] for log in log_history if "loss" in log]
    eval_loss = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epochs = list(range(1, len(eval_loss)+1))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss[:len(epochs)], label="Training Loss", marker='o')
    plt.plot(epochs, eval_loss, label="Validation Loss", marker='o')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_answer_length_distribution(df):
    lengths = [len(ans['text'][0].split()) if ans['text'] else 0 for ans in df['answers']]
    plt.hist(lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title("Answer Length Distribution")
    plt.xlabel("Number of Words in Answer")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_emotion_distribution(df):
    if 'emotion' not in df.columns:
        print("⚠️ 'emotion' column not found in DataFrame.")
        return
    emotion_counts = df['emotion'].value_counts()
    emotion_counts.plot(kind='bar', color='coral')
    plt.title("Emotion Label Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Number of QA Pairs")
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def postprocess_predictions(predictions, dataset, tokenizer):
    pred_answers = []
    true_answers = []

    for i, example in enumerate(dataset):
        inputs = tokenizer(
            example["question"],
            example["context"],
            truncation=True,
            padding="max_length",
            max_length=384,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        start_logits, end_logits = predictions
        start = torch.argmax(torch.tensor(start_logits[i]))
        end = torch.argmax(torch.tensor(end_logits[i])) + 1

        if start >= len(offset_mapping) or end >= len(offset_mapping):
            pred_answer = ""
        else:
            start_char = offset_mapping[start][0]
            end_char = offset_mapping[end][1]
            pred_answer = example["context"][start_char:end_char]

        pred_answers.append({'id': str(i), 'prediction_text': pred_answer})
        true_answers.append({'id': str(i), 'answers': example["answers"]})

    return pred_answers, true_answers

def evaluate_model(trainer, dataset, hf_dataset):
    squad_metric = evaluate.load("squad")
    predictions = trainer.predict(hf_dataset)
    pred_start, pred_end = predictions.predictions
    pred_answers, true_answers = postprocess_predictions((pred_start, pred_end), dataset, tokenizer)

    metrics = squad_metric.compute(predictions=pred_answers, references=true_answers)
    print("📊 Evaluation Metrics:", metrics)

    labels = list(metrics.keys())
    scores = list(metrics.values())
    plt.bar(labels, scores, color=['green', 'blue'])
    plt.title("QA Evaluation: EM & F1 Score")
    plt.ylabel("Score (%)")
    for i, val in enumerate(scores):
        plt.text(i, val + 1, f"{val:.2f}", ha='center')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

    return metrics

def infer(model, tokenizer, question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )
    print(f"\n🔍 Question: {question}\n📄 Answer: {answer.strip()}")

if __name__ == "__main__":
    train_data, val_data = load_squad("train-v1.1.json", "dev.csv")
    train_data = add_token_positions(train_data)
    val_data = add_token_positions(val_data)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_data),
        'validation': Dataset.from_pandas(val_data)
    })

    tokenized_ds = dataset.map(preprocess)

    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    # ✅ Fixed TrainingArguments for older versions
    training_args = TrainingArguments(
        output_dir="./results",
        do_eval=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer
    )

    trainer.train()

    pd.DataFrame(trainer.state.log_history).to_csv("training_log.csv", index=False)

    plot_loss(trainer.state.log_history)
    plot_answer_length_distribution(train_data)
    plot_answer_length_distribution(val_data)
    plot_emotion_distribution(val_data)

    evaluate_model(trainer, val_data.to_dict(orient="records"), tokenized_ds["validation"])

    infer(model, tokenizer,
          "Who wrote the play Hamlet?",
          "William Shakespeare wrote many plays. One of them is Hamlet.")

    model.save_pretrained("final_qa_model")
    tokenizer.save_pretrained("final_qa_model")
    print("✅ Model and tokenizer saved to 'final_qa_model/'")
