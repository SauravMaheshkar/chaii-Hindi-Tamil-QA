__all__ = ["CONFIG"]

CONFIG = dict(
    # Model
    model_type="bert",
    model_name_or_path="google/muril-large-cased",
    config_name="google/muril-large-cased",
    output_head_dropout_prob=0.0,
    gradient_accumulation_steps=2,
    # Tokenizer
    tokenizer_name="google/muril-large-cased",
    max_seq_length=400,
    doc_stride=135,
    # Training
    epochs=1,
    folds=4,
    train_batch_size=2,
    eval_batch_size=8,
    # Optimizer
    optimizer_type="AdamW",
    learning_rate=1.5e-5,
    weight_decay=1e-2,
    epsilon=1e-8,
    max_grad_norm=1.0,
    # Scheduler
    decay_name="cosine-warmup",
    warmup_ratio=0.1,
    logging_steps=100,
    # Misc
    output_dir="output",
    seed=21,
    # W&B
    competition="chaii",
    _wandb_kernel="sauravm",
)
