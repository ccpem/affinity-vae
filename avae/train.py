import logging
import os

import lightning as lt
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard

from . import utils_learning, vis
from .cyc_annealing import setup_annealing
from .data import load_data
from .loss import AVAELoss
from .models import model_setup
from .utils import as_list, latest_file
from .utils_gpu import setup_gpus
from .utils_learning import (
    build_meta_df,
    combine_meta_df,
    configure_optimiser,
    log_progress,
)


def train(params):
    """Function to train an AffinityVAE model. The inputs are training configuration parameters. In this function the
    data is loaded, selected and split into training, validation and test sets, the model is initialised and trained
    over epochs, the results are evaluated visualised and saved and the epoch level with a frequency configured with
    input parameters.

    Parameters
    ----------
    params : object
        A pydantic validated object containing all the training configuration parameters as attributes.
    """

    lt.pytorch.seed_everything(42)

    # ############################### LOGGING #################################

    writer = (
        torch.utils.tensorboard.SummaryWriter() if params.tensorboard else None
    )

    # ############################### GPU SETUP ################################

    fabric = setup_gpus(
        gpu=params.gpu,
        gpu_devices=params.gpu_devices,
        strategy=params.strategy,
    )

    fabric.launch()
    device = fabric.device
    rank_zero = fabric.global_rank == 0

    # DDP intentionally stashes autograd nodes across streams; silence the
    # resulting benign AccumulateGrad stream-mismatch warning if available.
    if hasattr(
        torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"
    ):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    # ############################### ANNEALING ###############################

    beta_arr = setup_annealing(
        epochs=params.epochs,
        value_max=params.beta,
        value_min=params.beta_min,
        cyc_method=params.cyc_method_beta,
        n_cycle=params.beta_cycle,
        ratio=params.beta_ratio,
        cycle_load=params.beta_load,
    )

    gamma_arr = setup_annealing(
        epochs=params.epochs,
        value_max=params.gamma,
        value_min=params.gamma_min,
        cyc_method=params.cyc_method_gamma,
        n_cycle=params.gamma_cycle,
        ratio=params.gamma_ratio,
        cycle_load=params.gamma_load,
    )
    if rank_zero and params.vis_cyc:
        vis.plot_cyc_variable(beta_arr, "beta", vis_format=params.vis_format)
        vis.plot_cyc_variable(gamma_arr, "gamma", vis_format=params.vis_format)

    # ############################### DATA ###############################
    trains, vals, tests, affinity_matrix, data_dim = load_data(
        eval=False,
        datapath=params.datapath,
        datatype=params.datatype,
        lim=params.limit,
        splt=params.split,
        batch=params.batch,
        no_val_drop=params.no_val_drop,
        affinity_path=params.affinity,
        classes=params.classes,
        gaussian_blur=params.gaussian_blur,
        normalise=params.normalise,
        shift_min=params.shift_min,
        rescale=params.rescale,
        vis_his=params.vis_his,
        vis_aff=params.vis_aff,
        vis_format=params.vis_format,
        fabric=fabric,
    )

    if len(trains) == 0:
        error = (
            "Training size is 0. Check the 'batch_size' parameter if it's not larger than your existing data."
            " If you're running distributed learning, each GPU gets an equal fraction of the data, adjust your"
            " batch size accordingly. If you are using a small dataset, consider using a smaller number of GPUs"
            " or running on a single GPU. Finally check 'limit' parameter if it is not set to smaller than batch_size."
        )
        logging.error(error)
        raise RuntimeError(error)

    dshape = next(iter(trains))[0].shape[2:]
    pose = not (params.pose_dims == 0)

    # ############################### MODEL ###############################
    vae = model_setup(
        model_type=params.model,
        input_shape=dshape,
        channels=params.channels,
        depth=params.depth,
        latent_dims=params.latent_dims,
        pose_dims=params.pose_dims,
        bnorm_encoder=params.bnorm_encoder,
        bnorm_decoder=params.bnorm_decoder,
        n_splats=params.n_splats,
        gsd_conv_layers=params.gsd_conv_layers,
        device=device,
        filters=params.filters,
    )

    # ################################# LOSS #################################

    loss = AVAELoss(
        device=device,
        beta=beta_arr,
        gamma=gamma_arr,
        lookup_aff=affinity_matrix,
        recon_loss=params.recon_loss,
        klreduction=params.klreduction,
    )

    # ############################### OPTIMISER ###############################
    optimizer = configure_optimiser(
        opt_method=params.opt_method, model=vae, learning_rate=params.learning
    )

    vae, optimizer = fabric.setup(vae, optimizer)

    # ############################### RESTARTS ################################
    e_start = 0
    t_history = []
    v_history = []

    if params.restart:
        if state is None:
            if not os.path.exists("states"):
                raise RuntimeError(
                    "There are no existing model states saved or provided either via the state flag or in the config. Unable to evaluate."
                )
            else:
                state = latest_file("states", ".pt")
                state = os.path.join("states", state)

        checkpoint = torch.load(state, weights_only=False)
        vae.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        e_start = checkpoint["epoch"]
        t_history = checkpoint["t_loss_history"]
        v_history = checkpoint["v_loss_history"]

    # ########################## TRAINING LOOP ################################

    logging.info("\n")
    logging.info("############################################### TRAINING")

    for epoch in range(e_start, params.epochs):

        # populate loss with new epoch
        t_history.append(np.zeros(4))
        v_history.append(np.zeros(4))

        # Per-rank epoch metadata buffers grouped by mode.
        filename_train, filename_val, filename_test = [], [], []
        meta_train, meta_val, meta_test = [], [], []

        # create holders for data, labels, and latent spaces
        x_train, x_val, x_test = [], [], []
        xhat_train, xhat_val, xhat_test = [], [], []

        y_train, y_val, y_test = [], [], []
        z_train, z_val, z_test = [], [], []
        c_train, c_val, c_test = [], [], []

        if pose:
            p_train, p_val, p_test = [], [], []

        # ########################## TRAINING #################################
        vae.train()
        for batch_number, (x, ys, aff, meta_data) in enumerate(trains):

            # get data in the right device
            x, aff = x.to(device), aff.to(device)
            x = x.to(torch.float32)

            # forward
            x_hat, lat_mu, lat_logvar, lat, lat_pose = vae(x)
            history_loss = loss(
                x, x_hat, lat_mu, lat_logvar, epoch, batch_aff=aff
            )

            # record loss
            for i in range(len(t_history[-1])):
                t_history[-1][i] += history_loss[i].item()
            log_progress(
                "Epoch: [%d/%d] | Batch: [%d/%d]"
                % (epoch + 1, params.epochs, batch_number + 1, len(trains))
            )

            # backwards
            fabric.backward(history_loss[0])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            z_train.extend(lat_mu.cpu().detach().numpy())  # store latents
            y_train.extend(as_list(ys))
            c_train.extend(lat_logvar.cpu().detach().numpy())
            if pose:
                p_train.extend(lat_pose.cpu().detach().numpy())

            filename_train.extend(as_list(meta_data.get("filename", [])))
            meta_train.extend(as_list(meta_data.get("meta", [])))
            x_train.extend(as_list(meta_data.get("image", [])))
            xhat_train.extend(vis.format(x_hat, data_dim))

        t_history[-1] /= len(trains)

        logging.info(
            "Training : Epoch: [%d/%d] | Loss: %f | Recon: %f | "
            "KLdiv: %f | Affin: %f | Beta: %f | Gamma: %f"
            % (
                epoch + 1,
                params.epochs,
                *t_history[-1],
                beta_arr[epoch],
                gamma_arr[epoch],
            )
        )
        # ########################## VAL ######################################
        vae.eval()
        with torch.inference_mode():
            for batch_number, (v, ys, aff, meta_data) in enumerate(vals):

                # get data in the right device
                v, aff = v.to(device), aff.to(device)
                v = v.to(torch.float32)

                # forward
                v_hat, v_mu, v_logvar, vlat, vlat_pos = vae(v)
                v_history_loss = loss(
                    v, v_hat, v_mu, v_logvar, epoch, batch_aff=aff
                )

                # record loss
                for i in range(len(t_history[-1])):
                    v_history[-1][i] += v_history_loss[i].item()
                log_progress(
                    "Epoch: [%d/%d] | Batch: [%d/%d]"
                    % (epoch + 1, params.epochs, batch_number + 1, len(vals))
                )

                z_val.extend(v_mu.cpu().detach().numpy())  # store latents
                y_val.extend(as_list(ys))
                c_val.extend(v_logvar.cpu().detach().numpy())
                if pose:
                    p_val.extend(vlat_pos.cpu().detach().numpy())

                filename_val.extend(as_list(meta_data.get("filename", [])))
                meta_val.extend(as_list(meta_data.get("meta", [])))
                x_val.extend(as_list(meta_data.get("image", [])))
                xhat_val.extend(vis.format(v_hat, data_dim))
            v_history[-1] /= len(vals)

            logging.info(
                "Validation : Epoch: [%d/%d] | Loss: %f | Recon: %f | "
                "KLdiv: %f | Affin: %f | Beta: %f | Gamma: %f"
                % (
                    epoch + 1,
                    params.epochs,
                    *v_history[-1],
                    beta_arr[epoch],
                    gamma_arr[epoch],
                )
            )

            if writer:
                for i, loss_name in enumerate(
                    ["Loss", "Recon loss", "KLdiv loss", "Affin loss"]
                ):
                    writer.add_scalar(loss_name, v_history[-1][i], epoch)

            # ########################## TEST #####################################
            if params.freq_eval != 0 and (epoch + 1) % params.freq_eval == 0:
                for batch_number, (t, ys, aff, meta_data) in enumerate(
                    tests
                ):  # tests empty if no 'test' dir
                    # get data in the right device
                    t, aff = t.to(device), aff.to(device)
                    t = t.to(torch.float32)

                    # forward
                    t_hat, t_mu, t_logvar, tlat, tlat_pose = vae(t)

                    z_test.extend(t_mu.cpu().detach().numpy())  # store latents
                    y_test.extend(as_list(ys))
                    c_test.extend(t_logvar.cpu().detach().numpy())
                    if pose:
                        p_test.extend(tlat_pose.cpu().detach().numpy())

                    filename_test.extend(
                        as_list(meta_data.get("filename", []))
                    )
                    meta_test.extend(as_list(meta_data.get("meta", [])))
                    x_test.extend(as_list(meta_data.get("image", [])))
                    xhat_test.extend(vis.format(t_hat, data_dim))

                    log_progress(
                        "Epoch: [%d/%d] | Batch: [%d/%d]"
                        % (
                            epoch + 1,
                            params.epochs,
                            batch_number + 1,
                            len(tests),
                        )
                    )
                logging.info(
                    "Evaluation : Epoch: [%d/%d]" % (epoch + 1, params.epochs)
                )
            logging.info("\n")  # end of training round

        needs_meta_df = (
            (
                # visualising embedding
                params.vis_emb
                and (epoch + 1) % params.freq_emb == 0
            )
            or (
                # visualising dynamic embedding
                params.vis_dynamic
                and (epoch + 1) % params.freq_dynamic == 0
            )
            or (
                # saving states of the model
                params.freq_sta != 0
                and (epoch + 1) % params.freq_sta == 0
            )
        )
        if needs_meta_df:
            meta_df = build_meta_df(
                pose=pose,
                train={
                    "filename": filename_train,
                    "meta": meta_train,
                    "x": x_train,
                    "xhat": xhat_train,
                    "y": y_train,
                    "z": z_train,
                    "logvar": c_train,
                    "pose": p_train if pose else None,
                },
                val={
                    "filename": filename_val,
                    "meta": meta_val,
                    "x": x_val,
                    "xhat": xhat_val,
                    "y": y_val,
                    "z": z_val,
                    "logvar": c_val,
                    "pose": p_val if pose else None,
                },
                test={
                    "filename": filename_test,
                    "meta": meta_test,
                    "x": x_test,
                    "xhat": xhat_test,
                    "y": y_test,
                    "z": z_test,
                    "logvar": c_test,
                    "pose": p_test if pose else None,
                },
            )

            # Dynamic plots and save should use metadata from all ranks.
            combined_meta_df = combine_meta_df(
                meta_df=meta_df,
                rank_zero=rank_zero,
                world_size=fabric.world_size,
            )

        # ########################## VISUALISE ################################
        logging.info("")

        static_embedding_due = (
            rank_zero and params.vis_emb and (epoch + 1) % params.freq_emb == 0
        )
        dynamic_embedding_due = (
            rank_zero
            and params.vis_dynamic
            and (epoch + 1) % params.freq_dynamic == 0
        )
        if static_embedding_due or dynamic_embedding_due:
            latent_columns = [
                col for col in combined_meta_df if col.startswith("lat")
            ]
            embedding_xs = combined_meta_df[latent_columns].to_numpy()
            embedding_ys = combined_meta_df["id"].to_numpy().copy()
            embedding_ys[combined_meta_df["mode"].to_numpy() == "tst"] = "test"
            latent_embedding = utils_learning.tsne_embedding(embedding_xs)
            if static_embedding_due and pose:
                pose_columns = [
                    col for col in combined_meta_df if col.startswith("pos")
                ]
                pose_xs = combined_meta_df[pose_columns].to_numpy()
                pose_embedding = utils_learning.tsne_embedding(pose_xs)
            logging.info("")

        # visualise accuracy: confusion and F1 scores
        if (
            params.vis_acc
            and params.freq_acc != 0
            and (epoch + 1) % params.freq_acc == 0
        ):
            combined_accuracy_data = utils_learning.combine_accuracy_data(
                z_train,
                y_train,
                z_val,
                y_val,
                rank_zero=rank_zero,
                world_size=fabric.world_size,
            )

            if rank_zero:
                assert combined_accuracy_data is not None
                (
                    z_train_all,
                    y_train_all,
                    z_val_all,
                    y_val_all,
                ) = combined_accuracy_data
                (
                    train_acc,
                    val_acc,
                    _,
                    ypred_train,
                    ypred_val,
                ) = utils_learning.accuracy(
                    z_train_all,
                    y_train_all,
                    z_val_all,
                    y_val_all,
                    classifier=params.classifier,
                )

                logging.info(
                    "------------------->>> Accuracy: Train: %f | Val: %f\n"
                    % (train_acc, val_acc),
                )
                vis.accuracy_plot(
                    y_train_all,
                    ypred_train,
                    y_val_all,
                    ypred_val,
                    epoch=epoch,
                    writer=writer,
                    vis_format=params.vis_format,
                )

                vis.f1_plot(
                    y_train_all,
                    ypred_train,
                    y_val_all,
                    ypred_val,
                    epoch=epoch,
                    writer=writer,
                    vis_format=params.vis_format,
                )

        # visualise loss
        if rank_zero and params.vis_los and epoch > 0:
            p = [
                len(trains),
                params.depth,
                params.channels,
                params.latent_dims,
                params.learning,
                beta_arr[epoch],
                gamma_arr[epoch],
            ]
            vis.loss_plot(
                epoch + 1,
                beta_arr[: epoch + 1],
                gamma_arr[: epoch + 1],
                t_history,
                v_history,
                p=p,
                vis_format=params.vis_format,
            )

        # visualise reconstructions - last batch
        if rank_zero and params.vis_rec and (epoch + 1) % params.freq_rec == 0:
            vis.recon_plot(
                x,
                x_hat,
                y_train,
                data_dim,
                mode="trn",
                epoch=epoch,
                writer=writer,
                vis_format=params.vis_format,
            )
            vis.recon_plot(
                v,
                v_hat,
                y_val,
                data_dim,
                mode="val",
                epoch=epoch,
                writer=writer,
                vis_format=params.vis_format,
            )

        # visualise mean and logvar similarity matrix
        if rank_zero and params.vis_sim and (epoch + 1) % params.freq_sim == 0:
            vis.latent_space_similarity_plot(
                z_train,
                np.array(y_train),
                mode="_train",
                epoch=epoch,
                affinity_matrix=params.affinity,
                vis_format=params.vis_format,
            )
            vis.latent_space_similarity_plot(
                z_val,
                np.array(y_val),
                mode="_valid",
                epoch=epoch,
                affinity_matrix=params.affinity,
                vis_format=params.vis_format,
            )

        # visualise embeddings
        if static_embedding_due:
            vis.latent_embed_plot_tsne(
                embedding_xs,
                embedding_ys,
                epoch=epoch,
                writer=writer,
                vis_format=params.vis_format,
                embedding=latent_embedding,
            )

            if pose:
                vis.latent_embed_plot_tsne(
                    pose_xs,
                    embedding_ys,
                    epoch=epoch,
                    writer=writer,
                    mode="pose",
                    vis_format=params.vis_format,
                    embedding=pose_embedding,
                )

        if dynamic_embedding_due:
            # merge img and rec into one image for display in altair
            dynamic_meta_df = combined_meta_df.copy()
            dynamic_meta_df["image"] = dynamic_meta_df["image"].apply(
                vis.merge
            )
            vis.dyn_latentembed_plot(
                dynamic_meta_df, epoch, embedding=latent_embedding
            )

        # visualise latent disentanglement
        if rank_zero and params.vis_dis and (epoch + 1) % params.freq_dis == 0:
            if not pose:
                poses = None
            else:
                poses = p_train
            vis.latent_disentamglement_plot(
                dshape,
                z_train,
                vae,
                device,
                poses=p_train if pose else None,
                vis_format=params.vis_format,
            )

        # visualise pose disentanglement
        if (
            rank_zero
            and pose
            and params.vis_pos
            and (epoch + 1) % params.freq_pos == 0
        ):
            vis.pose_disentanglement_plot(
                dshape,
                z_train,
                p_train,
                vae,
                device,
                vis_format=params.vis_format,
            )

            if params.vis_pose_class is not None:
                vis.pose_class_disentanglement_plot(
                    dshape,
                    z_train,
                    y_train,
                    params.vis_pose_class,
                    p_train,
                    vae,
                    device,
                    vis_format=params.vis_format,
                )

        # visualise interpolations
        if rank_zero and params.vis_int and (epoch + 1) % params.freq_int == 0:
            if len(tests) != 0:
                xs = np.r_[z_train, z_val, z_test]
                ys = np.r_[y_train, y_val, np.ones(len(z_test))]
                if pose:
                    ps = np.r_[p_train, p_val, p_test]
                else:
                    ps = None
            else:
                xs = np.r_[z_train, z_val]
                ys = np.r_[y_train, y_val]
                if pose:
                    ps = np.r_[p_train, p_val]
                else:
                    ps = None

            if params.vis_z_n_int is not None:
                vis.latent_4enc_interpolate_plot(
                    dshape,
                    xs,
                    ys,
                    vae,
                    device,
                    params.vis_z_n_int,
                    poses=ps,
                    vis_format=params.vis_format,
                )

            vis.interpolations_plot(
                dshape,
                xs,
                ys,
                vae,
                device,
                poses=ps,  # do we need val and test here?
                vis_format=params.vis_format,
            )
        # ########################## SAVE STATE ###############################
        if params.freq_sta != 0 and (epoch + 1) % params.freq_sta == 0:
            if rank_zero:
                if not os.path.exists("states"):
                    os.mkdir("states")

                mname = (
                    "avae_"
                    + str(params.date_time_run)
                    + "_E"
                    + str(epoch)
                    + "_"
                    + str(params.latent_dims)
                    + "_"
                    + str(params.pose_dims)
                    + ".pt"
                )

                logging.info(
                    "################################################################"
                )

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": vae._original_module.state_dict(),
                        "model_class_object": vae._original_module,
                        "optimizer_state_dict": optimizer._optimizer.state_dict(),
                        "t_loss_history": t_history,
                        "v_loss_history": v_history,
                    },
                    os.path.join("states", mname),
                )
                logging.info(
                    f"Saved model state: {mname} for restarting and evaluation "
                )

                filename = (
                    "meta_"
                    + str(params.date_time_run)
                    + "_E"
                    + str(epoch)
                    + "_L"
                    + str(params.latent_dims)
                    + "_P"
                    + str(params.pose_dims)
                    + ".pkl"
                )
                combined_meta_df.to_pickle(os.path.join("states", filename))

                logging.info(f"Saved meta file : {filename} for evaluation \n")
        fabric.barrier()

    if writer:
        writer.flush()
        writer.close()
