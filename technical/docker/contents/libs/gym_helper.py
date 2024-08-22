import io
import base64
from IPython.display import HTML
import os
import gymnasium as gym
from stable_baselines3 import PPO
import torch
import numpy as np

def show_run(eval_env, model, obs=None, max_steps=1000):
    monitored_env = gym.experimental.wrappers.RecordVideoV0(eval_env, "/tmp/.gym-results/")    
    if obs is not None:
        obs,_ = eval_env.reset(options={"new_state":obs})
    else:
        obs,_ = monitored_env.reset()
    video_name = None
    for _ in range(max_steps):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = monitored_env.step(action)
        if video_name is None:
            video_name = monitored_env._video_name
        if done or truncated: break
    monitored_env.close()
    video = None
    with io.open('/tmp/.gym-results/%s.mp4' % video_name, 'r+b') as f:
        video = f.read()
    encoded = base64.b64encode(video)
    os.remove('/tmp/.gym-results/%s.mp4' % video_name)
    return HTML(data='''
        <video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
    .format(encoded.decode('ascii')))

class OnnxableActionPolicy(torch.nn.Module):
    def __init__(self, extractor, policy_net, action_net):
        super(OnnxableActionPolicy, self).__init__()
        #self.extractor = extractor
        self.policy_net = policy_net
        self.action_net = action_net

    def forward(self, observation):
        #input_hidden, _ = self.extractor(observation)
        policy_hidden = self.policy_net(observation)
        action = self.action_net(policy_hidden)
        return action

def extract_onnx(model, save_to):
    model.policy.to("cpu")
    onnxable_model = OnnxableActionPolicy(model.policy.features_extractor, model.policy.mlp_extractor.policy_net, model.policy.action_net)
    dummy_input = torch.randn(1, 2)
    torch.onnx.export(onnxable_model, dummy_input, save_to, opset_version=9)
    print(f"Model saved in {save_to}")