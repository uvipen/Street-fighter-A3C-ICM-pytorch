"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Curiosity-driven Exploration by Self-supervised Prediction for Street Fighter""")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args

# (224, 384, 3)
def test(opt):
    torch.manual_seed(123)
    env, num_states, num_actions = create_train_env(1, "{}/video.mp4".format(opt.output_path))
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c_street_fighter".format(opt.saved_path)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_street_fighter".format(opt.saved_path),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset(False, False, True))
    round_done, stage_done, game_done = False, False, True
    while True:
        if round_done or stage_done or game_done:
            h_0 = torch.zeros((1, 1024), dtype=torch.float)
            c_0 = torch.zeros((1, 1024), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, round_done, stage_done, game_done = env.step(action)
        state = torch.from_numpy(state)
        if round_done or stage_done:
            state = torch.from_numpy(env.reset(round_done, stage_done, game_done))
        if game_done:
            print("Game over")
            break

if __name__ == "__main__":
    opt = get_args()
    test(opt)
