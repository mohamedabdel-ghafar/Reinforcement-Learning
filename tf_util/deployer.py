from tensorflow import Session
from tensorflow import saved_model
from agents import RLAgnet
from os.path import join, relpath


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
