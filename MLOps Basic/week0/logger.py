#%% [markdown]

# ## Pytorch Lightning Logger
# Lightning supports the most popular logging frameworks (Tensorboard, Comet, etc ...)  <br>
# To use a logger, simply pass it into the **Trainer**.  <br>
# Lightning uses TensorBoard by default  
#   
# [Custom Logger](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html#make-a-custom-logger)
# %%
from pytorch_lightning import loggers

logger = loggers.TensorBoardLogger("logs/")
