from hybrid_sac_learn_3 import SACagent
from golf_env_sejong1 import GolfEnv
import tensorflow as tf

def main():
    max_episode_num = 15000
    env = GolfEnv()
    agent = SACagent(env)

    with tf.device('/gpu:0'):
        agent.train(max_episode_num)

    agent.plot_result()


if __name__=="__main__":
    main()