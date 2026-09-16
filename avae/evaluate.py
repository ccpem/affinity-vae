import logging
import os

import lightning as lt
import numpy as np
import pandas as pd
import torch

from . import utils_learning, vis
from .data import load_data
from .utils import as_list, latest_file
from .utils_gpu import setup_gpus
from .utils_learning import build_meta_df, log_progress


def evaluate(params):
    """Function for evaluating the model. Loads the data, model and runs the evaluation. Saves the results of the
    evaluation in the plot and latents directories.

    Parameters
    ----------
    params : object
        Pydantic validated config object containing all the parameters required for evaluation.
    """

    # ############################### GPU SETUP ################################

    eval_gpu_device = (
        params.gpu_devices.split(",")[0].strip() if params.gpu_devices else "0"
    )
    fabric = setup_gpus(
        gpu=params.gpu,
        gpu_devices=eval_gpu_device,
        strategy="auto",
    )

    fabric.launch()
    device = fabric.device

    # ############################### DATA ###############################
    tests, data_dim = load_data(
        eval=True,
        datapath=params.datapath,
        datatype=params.datatype,
        lim=params.limit,
        batch=params.batch,
        gaussian_blur=params.gaussian_blur,
        normalise=params.normalise,
        shift_min=params.shift_min,
        rescale=params.rescale,
        vis_his=params.vis_his,
        vis_format=params.vis_format,
        fabric=fabric,
    )

    # ############################### MODEL ###############################

    if not os.path.exists("states") and params.state is None:
        raise RuntimeError(
            "There are no existing model states saved or provided either via the state flag or in the config. Unable to evaluate."
        )

    if not os.path.exists("states") and params.meta is None:
        raise RuntimeError(
            "There are no existing meta files saved or provided either via the meta flag or in the config. Unable to evaluate."
        )

    if params.state is None:
        state = latest_file("states", ".pt")
        params.state = os.path.join("states", state)
        logging.warning(
            "No model state provided for evaluation. Using latest model state: {}".format(
                params.state
            )
        )

    if params.meta is None:
        metas = latest_file("states", ".pkl")
        params.meta = os.path.join("states", metas)
        logging.warning(
            "No meta file provided for evaluation. Using latest meta file: {}".format(
                params.meta
            )
        )

    logging.info("\n")
    logging.info("############################################### MODEL")
    logging.info("Loading meta from: {}".format(params.meta))
    meta_df = pd.read_pickle(params.meta)

    logging.info("Loading model from: {}".format(params.state))
    checkpoint = torch.load(params.state, weights_only=False)
    vae = checkpoint["model_class_object"]
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae = fabric.setup(vae)

    dshape = next(iter(tests))[0].shape[2:]
    pose_dims = int(
        os.path.basename(params.state).split(".")[0].split("_")[-1]
    )

    # ########################## EVALUATE ################################

    logging.info("\n")
    logging.info("############################################### EVALUATION")

    # create holders for latent spaces and labels
    filename_test, meta_test, x_test, xhat_test, y_test, = (
        [],
        [],
        [],
        [],
        [],
    )
    z_test, c_test = [], []
    p_test = None

    if pose_dims != 0:
        p_test = []

    log_progress("Batch: [0/%d]" % (len(tests)))

    vae.eval()
    for batch_number, (t, label, aff, meta_data) in enumerate(tests):

        # get data in the right device
        t, aff = t.to(device), aff.to(device)
        t = t.to(torch.float32)

        # forward
        t_hat, t_mu, t_logvar, tlat, tlat_pose = vae(t)

        z_test.extend(t_mu.cpu().detach().numpy())  # store latents
        c_test.extend(t_logvar.cpu().detach().numpy())
        # if labels are present save them otherwise save test
        try:
            y_test.extend(label)
        except IndexError:
            np.full(shape=len(t), fill_value="test")
        if tlat_pose is not None:
            p_test.extend(tlat_pose.cpu().detach().numpy())

        filename_test.extend(as_list(meta_data.get("filename", [])))
        meta_test.extend(as_list(meta_data.get("meta", [])))
        x_test.extend(as_list(meta_data.get("image", [])))
        xhat_test.extend(vis.format(t_hat, data_dim))

        log_progress("Batch: [%d/%d]" % (batch_number + 1, len(tests)))
    logging.info(
        "\nEvaluation batches complete: [%d/%d]"
        % (batch_number + 1, len(tests))
    )

    eval_meta_df = build_meta_df(
        pose=p_test is not None,
        eval_data={
            "filename": filename_test,
            "meta": meta_test,
            "x": x_test,
            "xhat": xhat_test,
            "y": y_test,
            "z": z_test,
            "logvar": c_test,
            "pose": p_test if p_test is not None else None,
        },
    )

    meta_df = pd.concat([meta_df, eval_meta_df], ignore_index=False)

    # ########################## VISUALISE ################################

    # load class list for visualisations
    if params.classes is not None:
        classes_list = pd.read_csv(params.classes).columns.tolist()
    else:
        classes_list = []

    embedding_due = params.vis_emb or params.vis_dynamic
    if embedding_due:
        latent_columns = [col for col in meta_df if col.startswith("lat")]
        embedding_xs = meta_df[latent_columns].to_numpy()
        embedding_ys = meta_df["id"].to_numpy()
        latent_embedding = utils_learning.tsne_embedding(embedding_xs)
        eval_mask = meta_df["mode"].to_numpy() == "evl"

    # visualise reconstructions - last batch
    if params.vis_rec:
        vis.recon_plot(
            t,
            t_hat,
            y_test,
            data_dim,
            mode="evl",
            vis_format=params.vis_format,
        )

    # visualise latent disentanglement
    if params.vis_dis:
        vis.latent_disentamglement_plot(
            dshape,
            z_test,
            vae,
            device,
            poses=p_test,
            mode="_eval",
            vis_format=params.vis_format,
        )

    # visualise pose disentanglement
    if pose_dims != 0 and params.vis_pos:
        vis.pose_disentanglement_plot(
            dshape,
            z_test,
            p_test,
            vae,
            device,
            mode="_eval",
            vis_format=params.vis_format,
        )

    if pose_dims != 0 and params.vis_pose_class:
        vis.pose_class_disentanglement_plot(
            dshape,
            z_test,
            y_test,
            params.vis_pose_class,
            p_test,
            vae,
            device,
            mode="_eval",
            vis_format=params.vis_format,
        )
    # visualise interpolations
    if params.vis_int:
        vis.interpolations_plot(
            dshape,
            z_test,
            np.ones(len(z_test)),
            vae,
            device,
            poses=p_test,
            mode="_eval",
            vis_format=params.vis_format,
        )

    # visualise embeddings
    if params.vis_emb:
        vis.latent_embed_plot_tsne(
            embedding_xs[eval_mask],
            embedding_ys[eval_mask],
            classes_list,
            "_eval",
            vis_format=params.vis_format,
            embedding=latent_embedding[eval_mask],
        )

    if params.vis_sim:
        vis.latent_space_similarity_plot(
            z_test,
            np.array(y_test),
            mode="_eval",
            vis_format=params.vis_format,
        )

    # ############################# Predict #############################
    # get training latent space from metadata for comparison and accuracy estimation
    latents_training = meta_df[meta_df["mode"] == "trn"][
        [col for col in meta_df if col.startswith("lat")]
    ].to_numpy()
    latents_training_id = meta_df[meta_df["mode"] == "trn"]["id"]

    if params.vis_dynamic:
        # merge img and rec into one image for display in altair
        dynamic_meta_df = meta_df.copy()
        dynamic_meta_df["image"] = dynamic_meta_df["image"].apply(vis.merge)
        vis.dyn_latentembed_plot(
            dynamic_meta_df,
            0,
            mode="_eval",
            embedding=latent_embedding,
        )

    # visualise embeddings
    if params.vis_emb:
        vis.latent_embed_plot_tsne(
            embedding_xs,
            embedding_ys,
            classes_list,
            "_train_eval_comparison",
            vis_format=params.vis_format,
            embedding=latent_embedding,
        )

    # visualise accuracy
    (
        train_acc,
        val_acc,
        val_acc_selected,
        ypred_train,
        ypred_val,
    ) = utils_learning.accuracy(
        latents_training,
        np.array(latents_training_id),
        z_test,
        np.array(y_test),
        classifier=params.classifier,
    )
    logging.info(
        "------------------->>> Accuracy: Train: %f | Val : %f | Val with unseen labels: %f\n"
        % (train_acc, val_acc_selected, val_acc)
    )
    vis.accuracy_plot(
        np.array(latents_training_id),
        ypred_train,
        y_test,
        ypred_val,
        params.classes,
        mode="_eval",
        vis_format=params.vis_format,
    )
    vis.f1_plot(
        np.array(latents_training_id),
        ypred_train,
        y_test,
        ypred_val,
        mode="_eval",
        vis_format=params.vis_format,
    )
    logging.info("Saving meta files with evaluation data.")

    metas = os.path.basename(params.meta)
    # save metadata with evaluation data
    if not os.path.exists("states"):
        os.makedirs("states")
    meta_df.to_pickle(
        os.path.join("states", metas.split(".")[0] + "_eval.pkl")
    )
