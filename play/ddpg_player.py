from play import Player
import tensorflow.compat.v1 as tf
import gym


class DDPGPlayer(Player):
    def __init__(self, frozen_graph_path, state_node_name, actor_output_name):
        with tf.gfile.GFile(frozen_graph_path, 'rb') as graph_file:
            graph_def = tf.GraphDef()
            graph_def.FromString(graph_file.read())
            print("here")
            for nodeDef in graph_def.node:
                print(nodeDef.name)
            print("here 2")
        self.state_ph = tf.placeholder(dtype=tf.float32)
        self.action_op = tf.import_graph_def(graph_def, input_map={state_node_name: self.state_ph},
                                             return_elements=actor_output_name)

    def play(self, state, **kwargs):
        sess: tf.Session = kwargs["sess"]
        return sess.run(self.action_op, feed_dict={
            self.state_ph: state
        })


def play(env: gym.Env, agent_path, show=True, num_eps=100):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        player = DDPGPlayer(agent_path, "placeholder:0", "sample_agent/local/actor/add")
        for _ in range(num_eps):
            r_t = 0
            done = False
            s = env.reset()
            while not done:
                action = player.play(state=s, sess=sess)
                if show:
                    env.render()
                s_p, r, done, _ = env.step(action)
                r_t += r
            print(r_t)


if __name__ == "__main__":
    from os.path import curdir, join
    agent_path = join(curdir, "..", "tf_util", "frozen", "test_model.bytes", "model0")
    play(gym.make("MountainCarContinuous-v0"), agent_path)
