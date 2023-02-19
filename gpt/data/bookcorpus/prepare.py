from datasets import load_dataset

bookcorpus = load_dataset("bookcorpus", num_proc=8)
imagenet = load_dataset("imagenet-1k", num_proc=8)