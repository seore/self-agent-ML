import argparse

from agent.agent import Agent
from game import make_env, get_env_spec
from utils.plot import plot

def run_train(env_name: str, model_file: str | None):
    spec = get_env_spec(env_name)
    game = make_env(env_name)

    agent = Agent(input_size=spec["input_size"], output_size=spec["output_size"])
    
    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    model_name = model_file if model_file else spec["default_model"]

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done: 
            game.reset()

            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(model_name)
            
            total_score += score
            mean_score = total_score / agent.n_games
            scores.append(score)
            mean_scores.append(mean_score)

            print(
                f"[{env_name}] Game {agent.n_games} | Score: {score} | "
                f"Record: {record} | Mean: {mean_score:.2f}"        
            )

            plot(scores, mean_scores)


def run_play(env_name: str, model_file: str | None):
    spec = get_env_spec(env_name)
    game = make_env(env_name)
    agent = Agent(input_size=spec["input_size"], output_size=spec["output_size"])

    model_name = model_file if model_file else spec["default_model"]
    agent.load_trained(model_name)

    print("Play-only mode | env={env_name} | model=models/{model_name} (close window to stop)")

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)

        if done:
            print(f"[{env_name}] Game over! Score: {score}")
            game.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=["snake", "flappy"],
        default="snake",
        help="Which environment to run",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "play"],
        default="train",
        help="Train or watch the agent",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model filename inside models/ ",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.env, args.model)
    else:
        run_play(args.env, args.model)