import os
import numpy as np
import random
import time
import requests  # Ensure requests library is imported
from requests.exceptions import RequestException  # Import base class for request exceptions

import swanlab


import argparse
import sys
import torch
import signal
from os.path import dirname, abspath
from envs import REGISTRY as env_REGISTRY
from baselines import REGISTRY as agent_REGISTRY
from runner import REGISTRY as runner_REGISTRY
from modules.multi_processing import MultiPeocessRunner
from modules.multi_processing_double import MultiPeocessRunnerDouble
from configs.utils import get_config, recursive_dict_update, signal_handler, merge_dict

import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def main(args):
    global swanlab_available

    default_config = get_config('experiment')
    env_config = get_config(args.env, 'envs')
    agent_config = get_config(args.agent, 'agents')



    if args.seed == None:
        args.seed = np.random.randint(0, 10000)
    if args.agent == 'tiecomm':
        args.block = 'no'
    elif args.agent in ['tiecomm_wo_inter', 'hiercomm_vae_co_wo_inter']:
        args.block = 'inter'
    elif args.agent in ['tiecomm_wo_intra', 'hiercomm_vae_co_wo_intra']:
        args.block = 'intra'

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    env_config['seed'] = args.seed

    # update configs
    exp_config = recursive_dict_update(default_config, vars(args))
    env_config = recursive_dict_update(env_config, vars(args))
    agent_config = recursive_dict_update(agent_config, vars(args))

    # merge config
    config = {}
    config.update(default_config)
    config.update(env_config)
    config.update(agent_config)

    use_swanlab = args.use_swanlab if hasattr(args, 'use_swanlab') else default_config.get('use_swanlab', False)
    # SwanLab
    swanlab_available = False
    if use_swanlab:
        try:
            swanlab_api_key = args.swanlab_api_key if hasattr(args, 'swanlab_api_key') else default_config.get(
                'swanlab_api_key', 's3KnOyNpiJ8RiAjzFPDrE')
            swanlab_project = args.swanlab_project if hasattr(args, 'swanlab_project') else default_config.get(
                'swanlab_project', 'sl2')

            swanlab.login(api_key=swanlab_api_key)
            swanlab_available = True
            print(f"SwanLab initialized with project: {swanlab_project}")
        except RequestException as e:
            print(f"SwanLab network error: {e}, will skip")
        except Exception as e:
            print(f"SwanLab login failed: {e}, will skip")
    else:
        print("SwanLab logging is disabled")

    # ======================================load config==============================================
    use_cuda = default_config.get('use_cuda', False)
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    # update
    default_config['device'] = device
    env_config['device'] = device
    agent_config['device'] = device

    exp_config['device'] = device
    env_config['device'] = device
    agent_config['device'] = device

    args = argparse.Namespace(**config)
    # ======================================swanlab==============================================
    results_path = os.path.join(dirname(abspath(__file__)), "results")
    args.exp_id = f"{args.map}_{args.agent}_{args.block}_m{args.moduarity}_v{args.vib}_{args.note}"
    tags = ['Ming', args.env, args.map, args.agent, args.memo]
    if swanlab_available:
        try:
            swanlab.init(
                project='sl2',
                experiment_name=args.exp_id,
                tags=tags,
                save_dir=results_path
            )
            swanlab.config.update(args)
        except RequestException as e:
            print(f"SwanLab initialization network error: {e}, will skip subsequent operations")
            swanlab_available = False  # Mark as unavailable, no further attempts
        except Exception as e:
            print(f"SwanLab initialization failed: {e}, will skip subsequent operations")
            swanlab_available = False


    # ======================================register environment==============================================
    env = env_REGISTRY[args.env](env_config)

    env_info = env.get_env_info()
    agent_config['obs_shape'] = env_info["obs_shape"]
    agent_config['n_actions'] = env_info["n_actions"]
    agent_config['n_agents'] = env_info["n_agents"]
    exp_config['episode_length'] = env_info["episode_length"]
    exp_config['n_agents'] = env_info["n_agents"]
    exp_config['n_actions'] = env_info["n_actions"]

    agent = agent_REGISTRY[args.agent](agent_config)
    if args.agent == 'ic3net':
        exp_config['hard_attn'] = True
        exp_config['commnet'] = True
        exp_config['detach_gap'] = 10
        exp_config['comm_action_one'] = True
    elif args.agent == 'commnet':
        exp_config['hard_attn'] = False
        exp_config['commnet'] = True
        exp_config['detach_gap'] = 10
    elif args.agent == 'tarmac':
        exp_config['hard_attn'] = False
        exp_config['commnet'] = True
        exp_config['detach_gap'] = 10
    elif args.agent == 'magic':
        exp_config['hard_attn'] = False
        exp_config['hid_size'] = 64
        exp_config['detach_gap'] = 10
    elif args.agent in ['ac_att', 'ac_att_noise']:
        exp_config['att_head'] = args.att_head
        exp_config['hid_size'] = args.hid_size
    elif args.agent in ['tiecomm', 'tiecomm_g', 'tiecomm_random', 'tiecomm_default']:
        exp_config['interval'] = agent_config['time_interval']
    elif args.agent in ['hiercomm_vae', 'hiercomm_random', 'hiercomm_com', 'hiercomm_coo']:
        pass
    else:
        pass

    # swanlab.watch(agent)

    epoch_size = exp_config['epoch_size']
    batch_size = exp_config['batch_size']
    run = runner_REGISTRY[args.agent]
    if args.use_multiprocessing:
        for p in agent.parameters():
            p.data.share_memory_()
        runner = MultiPeocessRunner(exp_config, lambda: run(exp_config, env, agent))
    else:
        runner = run(exp_config, env, agent)

    total_num_episodes = 0
    total_num_steps = 0

    for epoch in range(1, args.total_epoches + 1):
        epoch_begin_time = time.time()

        log = {}
        for i in range(epoch_size):
            batch_log = runner.train_batch(batch_size)
            merge_dict(batch_log, log)
            # print(i,batch_log['success'])

        total_num_episodes += log['num_episodes']
        total_num_steps += log['num_steps']

        # print('episode_return',(log['episode_return']/log['num_episodes']))

        epoch_time = time.time() - epoch_begin_time
        if swanlab_available:
            try:
                swanlab.log({'epoch': epoch,
                           'episode': total_num_episodes,
                           'epoch_time': epoch_time,
                           'total_steps': total_num_steps,
                           'episode_return': log['episode_return'] / log['num_episodes'],
                           "episode_steps": np.mean(log['episode_steps']),
                           'action_loss': log['action_loss'],
                           'value_loss': log['value_loss'],
                           'total_loss': log['total_loss'],
                           })

                if args.agent in ['tiecomm', 'tiecomm_wo_inter', 'tiecomm_wo_intra']:
                    swanlab.log({'epoch': epoch,
                               'god_action_loss': log['god_action_loss'],
                               'god_value_loss': log['god_value_loss'],
                               'god_total_loss': log['god_total_loss'],
                               'num_groups': log['num_groups'] / log['num_episodes'],
                               })


                elif args.agent in ['teamcomm']:
                    swanlab.log({'epoch': epoch,
                               'team_action_loss': log['team_action_loss'],
                               'team_value_loss': log['team_value_loss'],
                               'team_total_loss': log['team_total_loss'],
                               'num_groups': log['num_groups'] / log['num_episodes'],
                               })
                    if args.vib:
                        swanlab.log({'epoch': epoch,
                                   'vib_loss': log['vib_loss'],
                                   })

                    if args.moduarity:
                        swanlab.log({'epoch': epoch,
                                   'moduarity_loss': log['moduarity_loss'],
                                   })

                elif args.agent in ['tiecomm_default', 'teamcomm_random']:
                    swanlab.log({'epoch': epoch,
                               'num_groups': log['num_groups'] / log['num_episodes'],
                               })

                else:
                    pass

                if args.env == 'lbf':
                    swanlab.log({'epoch': epoch,
                               'episode': total_num_episodes,
                               'num_collisions': log['num_collisions'] / log['num_episodes'],
                               })

                if args.env == 'tj':
                    swanlab.log({'epoch': epoch,
                               'episode': total_num_episodes,
                               'success':log['success'],
                               })
                if sys.flags.interactive == 0 and args.use_multiprocessing:
                    runner.quit()

            except RequestException as e:
                print(f"SwanLab logging network error: {e}, stopping logging")
                swanlab_available = False
            except Exception as e:
                print(f"SwanLab logging failed: {e}, stopping logging")
                swanlab_available = False

    print('current epoch: {}/{}'.format(epoch, args.total_epoches))
    print("=====Done!!!=====")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HierComm')

    parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--groups', type=int, nargs='+', default=[3, 3], help='just for ac_att_noise')
    parser.add_argument('--env', type=str, default='mpe', help='environment name',
                        choices=['mpe', 'lbf', 'rware', 'tj'])
    parser.add_argument('--map', type=str, default="mpe-large-spread-v2", help='environment map name',
                        choices=['easy', 'medium', 'hard', 'mpe-large-spread-v1', 'mpe-large-spread-v2',
                                 'mpe-large-spread-v3', 'mpe-large-spread-v4', 'mpe-large-spread-v5',
                                 'mpe-large-spread-v6',
                                 'mpe-simple-tag-v1', 'Foraging-easy-v0', 'Foraging-medium-v0', 'Foraging-hard-v0',
                                 'Foraging-tiny-v0', 'Foraging-small-v0', 'Foraging-medium1-v0', 'Foraging-standard-v0',
                                 'Foraging-large-v0', 'Foraging-huge-v0',
                                 'Foraging-large_coop-v0',
                                 'Foraging-huge_coop-v0', 'Foraging-massive-v0', 'Foraging-extreme-v0',
                                 'Foraging-strategic_12-v0',
                                 'Foraging-strategic_15-v0', 'Foraging-tactical_18-v0', 'Foraging-coordinated_21-v0'])
    parser.add_argument('--time_limit', type=int, default=100, help='time limit')

    # Agent settings
    parser.add_argument('--agent', type=str, default="tiecomm", help='algorithm name',
                        choices=['teamcomm',
                                 'teamcomm_random',
                                 'tiecomm',
                                 'ac_att', 'ac_att_noise', 'ac_mlp', 'gnn',
                                 'commnet', 'ic3net', 'tarmac', 'magic'])
    parser.add_argument('--block', type=str, default="no", help='block type', choices=['no', 'inter', 'intra'])
    parser.add_argument('--moduarity', type=bool, default=False, help='use moduarity_loss')
    parser.add_argument('--vib', type=bool, default=True, help='use vib_loss')

    # Experiment settings
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    # SwanLab 设置
    parser.add_argument('--use_swanlab', action='store_true', default=True, help='Enable SwanLab logging')
    parser.add_argument('--swanlab_project', type=str, default='sl1', help='SwanLab project name')
    parser.add_argument('--swanlab_api_key', type=str, default='s3KnOyNpiJ8RiAjzFPDrE', help='SwanLab API key')

    parser.add_argument('--use_multiprocessing', action='store_true', help='use multiprocessing')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--total_epoches', type=int, default=1000, help='total number of training epochs')
    parser.add_argument('--n_processes', type=int, default=6, help='number of processes')
    parser.add_argument('--att_head', type=int, default=2, help='number of attention heads')
    parser.add_argument('--hid_size', type=int, default=64, help='hidden size')
    parser.add_argument('--note', type=str, default="ap", help='note')

    parser.add_argument('--core_selection_method', type=str, default='hybrid',
                        choices=['original', 'key_node_only', 'hybrid'],
                        help='Team core node selection method')

    parser.add_argument('--hybrid_weight', type=float, default=0.5,
                        help='Weight for key nodes in hybrid core selection')
    parser.add_argument('--enhance_key_node_intra_obs', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help='enhance_key_node_intra_obs')
    parser.add_argument('--key_node_intra_enhancement', type=float, default=1.5,
                        help='key_node_intra_enhancement')

    args = parser.parse_args()

    training_begin_time = time.time()
    signal.signal(signal.SIGINT, signal_handler)
    main(args)
    training_time = time.time() - training_begin_time
    print('training time: {} h'.format(training_time / 3600))
