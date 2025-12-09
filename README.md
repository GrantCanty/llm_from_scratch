# Project Details  
This project lets you create an LLM based off of the GPT2 architecture. This is a decoder only LLM. This package prioritizes ease of use in creating a custom decoder.  

## Functionality  
The project lets you create a decoder with the ability to change the following parameters:  
* Number of layers  
* Number of heads  
* Embedding dimensions  

The project has the following functions:  
* `train(epoch_count, optional: eval_freq, optional: eval_iter, optional: start_context, optional: drop_rate)`  
* `generate(text, optional: max_tokens, optional: temperature, optional: top_k)`  
* `save_weights`  
* `load_weights(path)`  

Create an instance of the package like this:  

```
import decoder

model = decoder.Decoder()
model.train(5, drop_rate=.25)
```