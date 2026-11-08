Here's a small experiment to measure different metrics which matter as benchmark of model's output
Here are few observations
Time to first token increases as a result of increase in number of prompt tokens
Intermediatry Time between tokens pretty much remained the same with increase in input prompt
End to end token time increases noticebaly with increase in input prompt, but the scale of increase between TFFT and End to end time varies(TFFT increased by more than 2x when input size is doubled but E2E time increased by less than 1.5x)
Keeping the input tokens constant, but increasing the output tokens to 4x seems to have increased the E2E latency 4x.
Why tokens/second alone can conceal poor user experience.
Tokens/ second doesn't really go into the fine granularity of the things which actually matter to the end user. The end user most of the time cares about how fast he gets teh result back. So, just like we saw in the previous step, Time to first token and Time between tokens might remain the same, but the end user may still end up having bad experience because the bottleneck is the the decode phase and that causes and that is not captured in tokens per second metric.


