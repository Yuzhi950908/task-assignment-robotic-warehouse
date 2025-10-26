import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import gymnasium as gym
import pandas as pd
#from tarware.heuristic import heuristic_episode
from tarware.heuristic_DQN_new import heuristic_episode


parser = ArgumentParser(description="Run tests with vector environments on WarehouseEnv", formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument(
        "--num_episodes",
        default=100,
        type=int,
        help="The seed to run with"
    )
parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="The seed to run with"
    )

parser.add_argument(
        "--render",
        action='store_true',
    )

args = parser.parse_args()

def info_statistics(infos, global_episode_return, episode_returns):
    _total_deliveries = 0
    _total_clashes = 0
    _total_stuck = 0
    for info in infos:
        _total_deliveries += info["shelf_deliveries"]
        _total_clashes += info["clashes"]
        _total_stuck += info["stucks"]
        info["total_deliveries"] = _total_deliveries
        info["total_clashes"] = _total_clashes
        info["total_stuck"] = _total_stuck
    last_info = infos[-1]
    last_info["episode_length"] = len(infos)
    last_info["global_episode_return"] = global_episode_return
    last_info["episode_returns"] = episode_returns
    return last_info

if __name__ == "__main__":

    env = gym.make("tarware-extralarge-14agvs-7pickers-partialobs-v1")
    seed = args.seed
    completed_episodes = 0

    # 新增：创建空的 DataFrame
    results_df = pd.DataFrame(columns=['episode_id', 'overall_pick_rate', 'global_return', 'total_deliveries', 'total_clashes', 'total_stuck', 'episode_length', 'FPS'])

    for i in range(args.num_episodes):
        start = time.time()
        infos, global_episode_return, episode_returns = heuristic_episode(env.unwrapped, args.render, seed+i)
        end = time.time()
        last_info = info_statistics(infos, global_episode_return, episode_returns)
        last_info["overall_pick_rate"] = last_info.get("total_deliveries") * 3600 / (5 * last_info['episode_length'])
        episode_length = len(infos)
        print(f"Completed Episode {completed_episodes}: | [Overall Pick Rate={last_info.get('overall_pick_rate'):.2f}]| [Global return={last_info.get('global_episode_return'):.2f}]| [Total shelf deliveries={last_info.get('total_deliveries'):.2f}]| [Total clashes={last_info.get('total_clashes'):.2f}]| [Total stuck={last_info.get('total_stuck'):.2f}] | [FPS = {episode_length/(end-start):.2f}]")
        completed_episodes += 1

        # 新增：将数据添加到 DataFrame
        results_df = results_df.append({
        'episode_id': completed_episodes,
        'overall_pick_rate': last_info["overall_pick_rate"],
        'global_return': last_info["global_episode_return"],
        'total_deliveries': last_info["total_deliveries"],
        'total_clashes': last_info["total_clashes"],
        'total_stuck': last_info["total_stuck"],
        'episode_length': episode_length,
        'FPS': episode_length/(end-start)
        }, ignore_index=True)

        # 新增：保存 DataFrame 到 CSV 文件
        results_df.to_csv('results1.csv', index=False)