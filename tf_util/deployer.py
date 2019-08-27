import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.graph_util import convert_variables_to_constants


def deploy_model(saved_model_dir, frozen_graph_save_path, output_node_names):
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        tf.keras.backend.set_learning_phase(0)
        tf.saved_model.loader.load(sess, ["serve"], saved_model_dir)
        with g.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()))
            graph_def = g.as_graph_def()
            for node in graph_def.node:
                node.device = ""
            frozen_graph_def = convert_variables_to_constants(sess, graph_def, output_node_names, freeze_var_names)
            tf.io.write_graph(frozen_graph_def, name="model0", logdir=frozen_graph_save_path, as_text=False)
    del g
    return frozen_graph_def


if __name__ == "__main__":
    from os.path import join, relpath
    from os import curdir
    saved_model_path = relpath(join(curdir, "..", "train", "models", "export", "sample_agent__ddpg_SavedModel"))
    deploy_model(saved_model_path, join(curdir, "frozen", "test_model.bytes"), ["sample_agent/local/actor/add"])
