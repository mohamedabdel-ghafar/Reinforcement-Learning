from tensorflow import Session
from tensorflow import saved_model
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
import tensorflow as tf
from agents import RLAgnet
from os.path import join, relpath


def export_model(input_nodes_names, output_node_name):
    g1 = tf.Graph()
    with tf.Session(graph=g1) as sess:
        GRAPH_NAME = "cnn_model"
        tf.keras.backend.set_learning_phase(0)
        freeze_graph(input_saved_model_dir='/home/mohamed/PycharmProjects/RecModelAgora/models/saved.h5',
                     output_graph='./models/frozen_'+GRAPH_NAME+'.bytes',
                     output_node_names=output_node_name, clear_devices=True, input_saver=None,
                     input_binary=False, filename_tensor_name=None, initializer_nodes="", input_graph=None,
                     input_checkpoint=None, restore_op_name=None)
    g2 = tf.Graph()
    with tf.Session(graph=g2) as sess:
        input_graph_def = tf.GraphDef()
        with tf.gfile.Open('./models/frozen_' + GRAPH_NAME + '.bytes', 'rb') as f:
            input_graph_def.ParseFromString(f.read())
            output_graph_def = optimize_for_inference(
                input_graph_def, input_nodes_names, [output_node_name],
                tf.float32.as_datatype_enum)
        with tf.gfile.FastGFile('out/opt_' + GRAPH_NAME + '.bytes', "wb") as f:
            f.write(output_graph_def.SerializeToString())

            print("graph saved!")


# def save_frozen_graph(js_save_path, h5_save_path, deploy_path):
#     g = tf.Graph()
#     with tf.Session(graph=g) as sess:
#         tf.keras.backend.set_learning_phase(0)
#         to_deploy = model_from_json(open(js_save_path, 'r').read())
#         to_deploy.load_weights(h5_save_path)
#         with g.as_default():
#             freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()))
#             output_names = list([v.op.name for v in to_deploy.outputs])
#             print("outputs, ", output_names)
#             graph_def = g.as_graph_def()
#             for node in graph_def.node:
#                 node.device = ""
#             input_names = [v.op.name for v in to_deploy.inputs]
#             print(input_names)
#             frozen_graph_def = convert_variables_to_constants(sess, graph_def, output_names, freeze_var_names)
#             tf.io.write_graph(frozen_graph_def, name="model0", logdir=deploy_path, as_text=False)
#     del g
#     return frozen_graph_def



def deploy_model(rl_agent:RLAgnet, sess: Session, save_path, env_id, model_v):
    status_tensor = rl_agent.get_status_ph()
    action_tensor = rl_agent.get_action_op()
    model_path = relpath(join(save_path, env_id, model_v))
    info_input = saved_model.build_tensor_info(status_tensor)
    info_output = saved_model.build_tensor_info(action_tensor)
    sig_def = saved_model.build_signature_def(inputs={"status": info_input},
                                              outputs={"action": info_output},
                                              method_name=saved_model.signature_constants.CLASSIFY_METHOD_NAME)

    builder = saved_model.Builder(model_path)
    builder.add_meta_graph_and_variables(sess, [saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                             saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                 sig_def
                                         })
    builder.save()
