import io
import base64
from IPython.display import HTML
import os
import gymnasium as gym
from stable_baselines3 import PPO
import torch
import numpy as np
import sys

from torch import nn
import onnxruntime as ort

class NoOutput(object):
    def __init__(self,active=True):
        self.active = active
    
    def __enter__(self):
        if self.active:
            self.out = sys.stdout
            self.err = sys.stderr
            self.null = open(os.devnull, "w")
            sys.stdout = self.null
            sys.stderr = self.null

    def __exit__(self, type, value, traceback):
        if self.active:
            sys.stdout = self.out
            sys.stderr = self.err
            self.null.close()
            self.null = None

def show_run(eval_env, model, obs=None, max_steps=1000):
    with NoOutput(active=True):
        monitored_env = gym.wrappers.RecordVideo(eval_env, "/tmp/.gym-results/")  
        monitored_env.unwrapped._seed(42)
        if obs is not None:
            obs,_ = monitored_env.reset(options={"new_state":obs})
        else:
            obs,_ = monitored_env.reset()
        video_name = None
        for _ in range(max_steps):
            action, _states = model(obs,deterministic=True)
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
        <video style="width=100%" alt="test" controls autoplay><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
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


def get_env1():
    # Initialize Training Environment for ACC
    env = gym.make("acc-discrete-v0")
    # Invert the reward function...
    env.unwrapped.invert_loss=True
    env.unwrapped._seed(42)
    return env


def train_model1(env):
    # Initialize Agent
    architecture = dict(pi=[8], vf=[8])
    bad_model = PPO("MlpPolicy", env, verbose=1,policy_kwargs={"activation_fn":nn.ReLU,"net_arch":architecture})
    print("Performing some training steps...")
    bad_model.learn(total_timesteps=2_500)
    extract_onnx(bad_model,"bad_nn.onnx")
    return bad_model

def load_good_model():
    ort_sess_orig = ort.InferenceSession("./good_nn.onnx")
    def inference(x, deterministic=True):
        return np.argmax(ort_sess_orig.run(None, {"onnx::Gemm_0": [x]})[0]), None
    return inference
