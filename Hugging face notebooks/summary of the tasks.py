# %% [markdown]
# ## Summary of tasks
# This page shows the most frequent use-cases when using the library. The models available allow for many different
# configurations and a great versatility in use-cases. The most simple ones are presented here, showcasing usage for
# tasks such as question answering, sequence classification, named entity recognition and others.

# These examples leverage auto-models, which are classes that will instantiate a model according to a given checkpoint,
# automatically selecting the correct model architecture. Please check the `AutoModel` documentation
# for more information. Feel free to modify the code to be more specific and adapt it to your specific use-case.

# In order for a model to perform well on a task, it must be loaded from a checkpoint corresponding to that task. These
# checkpoints are usually pre-trained on a large corpus of data and fine-tuned on a specific task. This means the
# following:

# - Not all models were fine-tuned on all tasks. If you want to fine-tune a model on a specific task, you can leverage
#   one of the *run_$TASK.py* scripts in the [examples](https://github.com/huggingface/transformers/tree/master/examples) directory.
# - Fine-tuned models were fine-tuned on a specific dataset. This dataset may or may not overlap with your use-case and
#   domain. As mentioned previously, you may leverage the [examples](https://github.com/huggingface/transformers/tree/master/examples) scripts to fine-tune your model, or you may
#   create your own training script.

# In order to do an inference on a task, several mechanisms are made available by the library:

# - Pipelines: very easy-to-use abstractions, which require as little as two lines of code.
# - Direct model use: Less abstractions, but more flexibility and power via a direct access to a tokenizer
#   (PyTorch/TensorFlow) and full inference capacity.

# Both approaches are showcased here.
# %%