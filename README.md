# torchstate
Package for managing torch Stateful objects. This package is just a PoC at the moment.

# State Server
StateServer allows you to serve the live state dict out of memory. This is useful for live-ckpt recovery or any sort of use case where one needs a way to obtain a live tensor from a process.

### Server
```python
state_server = StateServer(state_dict, host="0.0.0.0", port=1234)
```

### Client
```python
url = "zbserver://192.168.0.2:1234"
client = StateClient(url)
tensor = client.get_tensor('[model][model.layers.0.self_attn.q_weight]')
lr = client.get_float('[optimizer][param_groups][0][lr]')
```

# Roadmap
- [ ] Streaming out of CPU
- [ ] Pipelined casting
- [ ] Streaming out of CUDA device
