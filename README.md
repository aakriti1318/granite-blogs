# granite-blogs

Blog posts and companion notebooks about building with [IBM Granite](https://huggingface.co/ibm-granite) models.

## Posts

| Post | Notebook | What it shows |
|---|---|---|
| [One Model, Three Specialists: A Policy Assistant for XYZ Bank](granite-switch-mellea/xyz-bank-policy-assistant.md) | [`xyz-bank-policy-assistant.ipynb`](granite-switch-mellea/xyz-bank-policy-assistant.ipynb) | One Granite Switch model doing three jobs (safety check, answerability check, reply-rule check), each called as a Python function through [Mellea](https://github.com/generative-computing/mellea). |

XYZ Bank, its policy text and the people in the story are fictional.

## Running a notebook

The notebooks call a Granite Switch model served by vLLM, so you need a machine with a GPU.

1. Serve the model. The exact command is in each notebook's "Serve the model" step.
2. Install the client libraries: `pip install "mellea[switch]>=0.9" openai`. The `switch` extra is required for Mellea to load the model's built-in adapters.
3. Open the notebook and run the cells in order.

Use Mellea 0.9 or newer. Older versions, for example 0.3.0, don't include the `core` and `guardian` functions the notebook imports.

## Which Granite Switch model?

Use [`granite-switch-4.1-3b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) for the policy assistant. It bundles 12 adapters, including the three this notebook uses. The 8B and 30B Switch models bundle a different set (the 8B config lists only `guardian-core` and `decision`), so the answerability and requirement checks won't work on them.

## Status

The notebook code has been checked for syntax and notebook-format validity, and its logic has been exercised with the model calls stubbed out. It has **not** been run against a live Granite Switch model, because that needs a GPU. Run it on your own endpoint and check the output before you publish the blog's example results.
