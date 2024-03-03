# video2scenario

Forms a tree-like folder dataset with L parts at the lowest level.

Then recursively writes descriptions of scenes with Large Language Models and Image Captioning Models.

The lowest level clips are [captioned with VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA).

The descriptions are gathered in a list and then the LLM is asked to describe the overall scene. Then the process continutes until the top level.

Any OpenAI-like text completion model can be used for this. In my tests [Oobabooga's text generation webui](https://github.com/oobabooga/text-generation-webui) is used as the API endpoint.

User can also provide the master prompt to help the model and edit the resulting descriptions with a Gradio demo interface.

There is also an option to store the resulting corrected output for better fine-tuning the models, for example, using a LoRA.

The Gradio interface has a dropdown to select each description - clip pair, on each level.

---

The goal of this subproject is to make a DiffusionOverDiffusion dataset to train [InfiNet](https://github.com/kabachuha/InfiNet) and the future complex script-based text2video models with minimal human labeling efforts.
