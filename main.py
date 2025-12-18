import argparse
import random
import numpy as np
import torch
import yaml

from game import make_env, get_env_spec
from agent.agent import Agent
from utils.run_manager import RunManager
from utils.plot import plot


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str):
    name = name.lower()
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_paths(env_name: str, spec: dict, model_arg: str | None):
    best_name = spec["default_best"]
    latest_name = spec["default_latest"]

    if model_arg:
        # allow user to pass explicit filename
        return RunManager.model_path(model_arg), RunManager.model_path(model_arg)

    return RunManager.model_path(best_name), RunManager.model_path(latest_name)


def run_train(args, cfg):
    spec = get_env_spec(args.env)
    best_path, latest_path = resolve_model_paths(args.env, spec, args.model)

    game = make_env(args.env, render=not args.headless, fps=args.fps, shaping=(args.shaping == "on"))

    agent = Agent(
        input_size=spec["input_size"],
        output_size=spec["output_size"],
        hidden_size=cfg["training"]["hidden_size"],
        lr=cfg["training"]["lr"],
        gamma=cfg["training"]["gamma"],
        max_memory=cfg["training"]["max_memory"],
        batch_size=cfg["training"]["batch_size"],
        epsilon_start=cfg["training"]["epsilon_start"],
        epsilon_end=cfg["training"]["epsilon_end"],
        epsilon_decay_games=cfg["training"]["epsilon_decay_games"],
        target_update_steps=cfg["training"]["target_update_steps"],
        double_dqn=bool(cfg["training"]["double_dqn"]),
        device=args.device,
    )

    run = RunManager(args.env, config={
        "env": args.env,
        "mode": "train",
        "device": str(args.device),
        "headless": args.headless,
        "fps": args.fps,
        "shaping": args.shaping,
        "config": cfg,
    })

    scores = []
    mean_scores = []
    total_score = 0
    record = -10**9

    while True:
        state = game.get_state()
        action = agent.act(state, greedy=False)

        reward, done, score = game.play_step(action)
        next_state = game.get_state()

        agent.remember(state, action, reward, next_state, done)
        loss = agent.train_step()

        if done:
            agent.n_games += 1
            game.reset()

            total_score += score
            mean_score = total_score / agent.n_games
            scores.append(score)
            mean_scores.append(mean_score)

            eps = agent.epsilon()
            if agent.n_games % cfg["training"]["log_every_games"] == 0:
                print(f"[{args.env}] Ep {agent.n_games} | Score {score} | Mean {mean_score:.2f} | Eps {eps:.3f} | Loss {loss}")
            run.log(agent.n_games, score, mean_score, eps, loss)

            # checkpointing
            if score > record:
                record = score
                agent.save(best_path)

            if agent.n_games % cfg["training"]["save_every_games"] == 0:
                agent.save(latest_path)
                plot(scores, mean_scores, run.plot_path)


def run_play(args, cfg):
    spec = get_env_spec(args.env)
    best_path, latest_path = resolve_model_paths(args.env, spec, args.model)

    # auto-load best if exists; else latest; else message
    model_path = best_path if os.path.exists(best_path) else latest_path

    if not os.path.exists(model_path):
        print(f"No model found for env='{args.env}'. Expected one of:\n  {best_path}\n  {latest_path}\nRun training first.")
        return

    game = make_env(args.env, render=True, fps=args.fps, shaping=(args.shaping == "on"))

    agent = Agent(
        input_size=spec["input_size"],
        output_size=spec["output_size"],
        hidden_size=cfg["training"]["hidden_size"],
        lr=cfg["training"]["lr"],
        gamma=cfg["training"]["gamma"],
        max_memory=cfg["training"]["max_memory"],
        batch_size=cfg["training"]["batch_size"],
        epsilon_start=cfg["training"]["epsilon_start"],
        epsilon_end=cfg["training"]["epsilon_end"],
        epsilon_decay_games=cfg["training"]["epsilon_decay_games"],
        target_update_steps=cfg["training"]["target_update_steps"],
        double_dqn=bool(cfg["training"]["double_dqn"]),
        device=args.device,
    )
    agent.load(model_path)

    print(f"Play mode | env={args.env} | model={model_path}")
    while True:
        state = game.get_state()
        action = agent.act(state, greedy=True)
        _, done, score = game.play_step(action)
        if done:
            print(f"[{args.env}] Game over. Score: {score}")
            game.reset()


def run_eval(args, cfg):
    spec = get_env_spec(args.env)
    best_path, latest_path = resolve_model_paths(args.env, spec, args.model)
    model_path = best_path if os.path.exists(best_path) else latest_path

    if not os.path.exists(model_path):
        print(f"No model found for env='{args.env}'. Train first.")
        return

    game = make_env(args.env, render=False, fps=args.fps, shaping=(args.shaping == "on"))

    agent = Agent(
        input_size=spec["input_size"],
        output_size=spec["output_size"],
        hidden_size=cfg["training"]["hidden_size"],
        lr=cfg["training"]["lr"],
        gamma=cfg["training"]["gamma"],
        max_memory=cfg["training"]["max_memory"],
        batch_size=cfg["training"]["batch_size"],
        epsilon_start=cfg["training"]["epsilon_start"],
        epsilon_end=cfg["training"]["epsilon_end"],
        epsilon_decay_games=cfg["training"]["epsilon_decay_games"],
        target_update_steps=cfg["training"]["target_update_steps"],
        double_dqn=bool(cfg["training"]["double_dqn"]),
        device=args.device,
    )
    agent.load(model_path)

    scores = []
    for ep in range(1, args.episodes + 1):
        game.reset()
        done = False
        while not done:
            state = game.get_state()
            action = agent.act(state, greedy=True)
            _, done, score = game.play_step(action)
        scores.append(score)

    print(f"Eval | env={args.env} | episodes={args.episodes}")
    print(f"  avg={sum(scores)/len(scores):.2f}  best={max(scores)}  worst={min(scores)}")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["snake", "flappy"], default="snake")
    parser.add_argument("--mode", choices=["train", "play", "eval"], default="train")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="Explicit model filename in models/ (optional)")

    parser.add_argument("--headless", action="store_true", help="Train without rendering (faster)")
    parser.add_argument("--plot", action="store_true", help="(Reserved) plots saved automatically to runs/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--shaping", choices=["on", "off"], default="on")
    parser.add_argument("--episodes", type=int, default=50, help="Eval episodes")

    args = parser.parse_args()

    cfg = load_config(args.config)

    set_seed(args.seed)
    args.device = pick_device(args.device)

    if args.mode == "train":
        run_train(args, cfg)
    elif args.mode == "play":
        run_play(args, cfg)
    else:
        run_eval(args, cfg)